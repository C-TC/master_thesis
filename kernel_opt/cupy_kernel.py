import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import utils
from timeit import repeat
from cupyx.profiler import benchmark


@cpx.jit.rawkernel()
def VA_f_0(out_data, indices, indptr, H_tile_1, H_tile_2):
    """ Computes A * (H * H^T), but H_tile_2 is transposed inside the function
    
    Each block computes a subset of the rows of the output. Each thread computes a subset of the columns of the parent
    block's rows. Using grid-strided pattern.
    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            for k in range(H_tile_1.shape[1]):
                # cpx.jit.atomic_add(out_data, j, H_tile_1[rowNo, k] * H_tile_2[colNo, k])
                out_data[j] += H_tile_1[rowNo, k] * H_tile_2[colNo, k]


# GAT
""" Computes: 
        C(i, j) = l(i) + r(j)
        D = A * C
        E = leakyrelu(D)
"""
forward_lr_E_rowmax_kernel = cp.RawKernel(
    r'''
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ static float my_atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

extern "C" __global__
void gat_forward_lr_E_rowmax(float* out_data, float* out_row_max, int* indices, int* indptr, float* l, float* r, int num_rows) {
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = bid; i < num_rows; i += num_blocks) {
        for (int j = indptr[i] + tid; j < indptr[i + 1]; j += num_threads) {
            int rowNo = i;
            int colNo = indices[j];
            float tmp = l[rowNo] + r[colNo];
            if (tmp <= 0) {
                tmp = tmp * 0.2;
            }
            out_data[j] = tmp;
            atomicMaxFloat(&out_row_max[rowNo], tmp);
        }
    }
}

''', 'gat_forward_lr_E_rowmax')

@cpx.jit.rawkernel()
def GAT_f_1(out_data, row_sum, E_data, indices, indptr, row_max):
    """Softmax on each row of E part 1:
        exp(E(i,j) - rowmax(i))
        stores the sum of each row in row_sum
        stores the result in out_data

    """
    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            tmp = cp.exp(E_data[j] - row_max[rowNo])
            out_data[j] = tmp
            cpx.jit.atomic_add(row_sum, rowNo, tmp)


@cpx.jit.rawkernel()
def GAT_f_2(alpha_data, row_sum, indices, indptr):
    """Softmax on each row of E part 2:
        Alpha(i,j) / row_sum(i)
        stores the result in alpha_data
    """
    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            alpha_data[j] /= row_sum[rowNo]


@cpx.jit.rawkernel()
def GAT_b_0(out_data, row_dot, Alpha_data, dZ, M, indices, indptr):
    """ dAlpha = A * (dZ @ M.T) and row_dot(i) = dAlpha(i,:) @ Alpha(i,:).T
        stores the dAlpha in out_data
        stores the row_dot in row_dot
    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            for k in range(dZ.shape[1]):
                out_data[j] += dZ[rowNo, k] * M[colNo, k]
            cpx.jit.atomic_add(row_dot, rowNo, out_data[j] * Alpha_data[j])


@cpx.jit.rawkernel()
def GAT_b_1(dl, dr, l, r, Alpha_data, dAlpha_data, indices, indptr, row_dot):
    """ dE(i,:) = (dAlpha(i,:) - row_dot(i)) * Alpha(i,:)
        Recompute D(i,j) = l(i) + r(j)
        dD from E = leakyrelu(D)
        dl(i) = sum(dD(i,:)), dr(j) = sum(dD(:,j))
    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            dE = (dAlpha_data[j] - row_dot[rowNo]) * Alpha_data[j]
            D_recomp = l[rowNo] + r[colNo]
            if D_recomp <= 0:
                dE *= 0.2
            cpx.jit.atomic_add(dl, rowNo, dE)
            cpx.jit.atomic_add(dr, colNo, dE)


@cpx.jit.rawkernel()
def AGNN_f_0(out_data, indices, indptr, H_tile_1, H_tile_2, H_tile_1_norm,
             H_tile_2_norm):
    """ Computes A * ((H x H^T) / (||H|| x ||H||^T)

    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            tmp = 0.0
            for k in range(H_tile_1.shape[1]):
                tmp += (H_tile_1[rowNo, k] * H_tile_2[colNo, k])
            out_data[j] = tmp / (H_tile_1_norm[rowNo] * H_tile_2_norm[colNo])


@cpx.jit.rawkernel()
def AGNN_b_0(dC_out_data, dD_out_data, indices, indptr, dZ, M, H_tile_1,
             H_tile_2, n_tile_1, n_tile_2):
    """ Computes 
        dQ = A * (dZ @ M.T)
        dC = dQ / D
        dD = -C * dQ / D^2
        where (recompute C and D):
        C = H_tile_1 @ H_tile_2.T
        D = n_tile_1 @ n_tile_2.T
    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            dQ = 0.0
            C = 0.0
            D = n_tile_1[rowNo] * n_tile_2[colNo]
            for k in range(dZ.shape[1]):
                dQ += dZ[rowNo, k] * M[colNo, k]
            for k in range(H_tile_1.shape[1]):
                C += H_tile_1[rowNo, k] * H_tile_2[colNo, k]

            dC_out_data[j] = dQ / D
            dD_out_data[j] = -C * dQ / (D * D)

KERNELS_CUPY = {
    'VA_f_0': VA_f_0,
    'GAT_f_0': forward_lr_E_rowmax_kernel,
    'GAT_f_1': GAT_f_1,
    'GAT_f_2': GAT_f_2,
    'GAT_b_0': GAT_b_0,
    'GAT_b_1': GAT_b_1,
    'AGNN_f_0': AGNN_f_0,
    'AGNN_b_0': AGNN_b_0
}
