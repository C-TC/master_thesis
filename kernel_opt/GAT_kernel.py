import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import dace


FULL_MASK = 0xFFFFFFFF

""" Computes: 
        C(i, j) = l(i) + r(j)
        D = A * C
        E = leakyrelu(D)
        row_max[i] = max(E[i, :])
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


M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz = dace.symbol('nnz')


@dace.program
def lr_E_rowmax_dace(out_data: dace.float32[nnz], out_row_max: dace.float32[M], indices: dace.int32[nnz], indptr: dace.int32[M + 1], l: dace.float32[M], r: dace.float32[N]):
    out_data[:] = 0
    out_row_max[:] = np.NINF
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            colNo = indices[j]
            tmp = l[rowNo] + r[colNo]
            tmp = np.maximum(tmp, 0.2 * tmp)
            out_data[j] = tmp
            out_row_max[rowNo] = np.maximum(out_row_max[rowNo], tmp)


@cpx.jit.rawkernel()
def gat_forward_softmax_1(out_data, row_sum, E_data, indices, indptr, row_max):
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

@dace.program
def softmax_1_dace(out_data: dace.float32[nnz], row_sum: dace.float32[M], E_data: dace.float32[nnz], indices: dace.int32[nnz], indptr: dace.int32[M + 1], row_max: dace.float32[M]):
    out_data[:] = 0
    row_sum[:] = 0
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            tmp = np.exp(E_data[j] - row_max[rowNo])
            out_data[j] = tmp
            row_sum[rowNo] += tmp



@cpx.jit.rawkernel()
def gat_forward_softmax_2(alpha_data, row_sum, indices, indptr):
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

@dace.program
def softmax_2_dace(alpha_data: dace.float32[nnz], row_sum: dace.float32[M], indices: dace.int32[nnz], indptr: dace.int32[M + 1]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            alpha_data[j] /= row_sum[rowNo]




@cpx.jit.rawkernel()
def gat_backward_dZ_dAlpha_rowdot(out_data, row_dot, Alpha_data, dZ, M,
                                  indices, indptr):
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
def gat_backward_dZ_dAlpha_rowdot_shfl(out_data, row_dot, Alpha_data, dZ, M,
                                       indices, indptr):
    """ dAlpha = A * (dZ @ M.T) and row_dot(i) = dAlpha(i,:) @ Alpha(i,:).T
        stores the dAlpha in out_data
        stores the row_dot in row_dot
    """

    bid = cpx.jit.blockIdx.x  # Block ID
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x  # Thread ID
    num_threads = cpx.jit.blockDim.x
    wid = tid // cpx.jit.warpsize  # Warp ID
    num_warps = cp.int32(cp.ceil(num_threads / cpx.jit.warpsize))
    twid = tid % cpx.jit.warpsize  # Thread ID within warp

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + wid, indptr[i + 1], num_warps):
            rowNo = i
            colNo = indices[j]
            dAlpha = 0.0
            for k in range(twid, dZ.shape[1], cpx.jit.warpsize):
                dAlpha += dZ[rowNo, k] * M[colNo, k]
            dAlpha += cpx.jit.shfl_down_sync(FULL_MASK, dAlpha, 16)
            dAlpha += cpx.jit.shfl_down_sync(FULL_MASK, dAlpha, 8)
            dAlpha += cpx.jit.shfl_down_sync(FULL_MASK, dAlpha, 4)
            dAlpha += cpx.jit.shfl_down_sync(FULL_MASK, dAlpha, 2)
            dAlpha += cpx.jit.shfl_down_sync(FULL_MASK, dAlpha, 1)
            if twid == 0:
                out_data[j] = dAlpha
                cpx.jit.atomic_add(row_dot, rowNo, out_data[j] * Alpha_data[j])


@dace.program
def dZ_dAlpha_rowdot_dace(out_data: dace.float32[nnz], row_dot: dace.float32[M], Alpha_data: dace.float32[nnz], dZ: dace.float32[M, K], M_mat: dace.float32[M, K], indices: dace.int32[nnz], indptr: dace.int32[M + 1]):
    out_data[:] = 0
    row_dot[:] = 0
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            colNo = indices[j]
            for k in dace.map[0:K]:
                out_data[j] += dZ[rowNo, k] * M_mat[colNo, k]
            row_dot[rowNo] += out_data[j] * Alpha_data[j]
    

@cpx.jit.rawkernel()
def gat_backward_softmax_leakyrelu_lr(dl, dr, l, r, Alpha_data, dAlpha_data,
                                      indices, indptr, row_dot):
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


@dace.program
def softmax_leakyrelu_lr_dace(dl: dace.float32[M], dr: dace.float32[N], l: dace.float32[M], r: dace.float32[N], Alpha_data: dace.float32[nnz], dAlpha_data: dace.float32[nnz], indices: dace.int32[nnz], indptr: dace.int32[M + 1], row_dot: dace.float32[M]):
    dl[:] = 0
    dr[:] = 0
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            colNo = indices[j]
            dE = (dAlpha_data[j] - row_dot[rowNo]) * Alpha_data[j]
            D_recomp = l[rowNo] + r[colNo]
            dE = np.where(D_recomp <= 0, dE * 0.2, dE)
            dl[rowNo] += dE
            dr[colNo] += dE
