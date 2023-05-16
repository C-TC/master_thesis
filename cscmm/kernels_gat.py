import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import utils
from timeit import repeat
from cupyx.profiler import benchmark

FULL_MASK = 0xFFFFFFFF

# TODO: optimize this kernel
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
            colNo = indices[j]
            alpha_data[j] /= row_sum[rowNo]


def test_gat_forward_kernels():
    # Test GAT forward gpu kernels
    print('Test GAT forward gpu kernels...')
    rng = np.random.default_rng(42)
    A = utils.generate_sparse_matrix(1000, 1000, 50000, np.float32, rng)
    A.data[:] = 1.0

    # Test gat_forward_lr_E_rowmax
    ref_E_data = np.zeros_like(A.data, dtype=np.float32)
    l = utils.generate_dense_matrix(A.shape[0], 1, np.float32, rng).squeeze()
    r = utils.generate_dense_matrix(A.shape[1], 1, np.float32, rng).squeeze()
    ref_row_max = np.full(A.shape[0], np.NINF, dtype=np.float32)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            ref_E_data[j] = l[i] + r[A.indices[j]]
            if ref_E_data[j] <= 0:
                ref_E_data[j] *= 0.2
            ref_row_max[i] = max(ref_row_max[i], ref_E_data[j])
    E_data_dev = cp.zeros_like(A.data, dtype=np.float32)
    row_max_dev = cp.full(A.shape[0], cp.NINF, dtype=cp.float32)
    forward_lr_E_rowmax_kernel(
        (min(65535,
             len(A.indptr) - 1), ), (128, ),
        (E_data_dev, row_max_dev, cp.asarray(A.indices), cp.asarray(
            A.indptr), cp.asarray(l), cp.asarray(r), len(A.indptr) - 1))

    assert cp.allclose(E_data_dev, ref_E_data)
    assert cp.allclose(row_max_dev, ref_row_max)

    # Test gat_forward_softmax_1
    ref_row_sum = np.zeros(A.shape[0], dtype=np.float32)
    ref_Alpha = np.zeros_like(A.data, dtype=np.float32)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            ref_Alpha[j] = np.exp(ref_E_data[j] - ref_row_max[i])
            ref_row_sum[i] += ref_Alpha[j]
    Alpha_dev = cp.zeros_like(A.data, dtype=cp.float32)
    row_sum_dev = cp.zeros(A.shape[0], dtype=cp.float32)
    gat_forward_softmax_1[min(65535,
                              len(A.indptr) - 1),
                          128](Alpha_dev, row_sum_dev, E_data_dev,
                               cp.asarray(A.indices), cp.asarray(A.indptr),
                               row_max_dev)

    assert cp.allclose(Alpha_dev, ref_Alpha)
    assert cp.allclose(row_sum_dev, ref_row_sum)

    # Test gat_forward_softmax_2
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            ref_Alpha[j] /= ref_row_sum[i]
    gat_forward_softmax_2[min(65535,
                              len(A.indptr) - 1), 128](Alpha_dev, row_sum_dev,
                                                       cp.asarray(A.indices),
                                                       cp.asarray(A.indptr))

    assert cp.allclose(Alpha_dev, ref_Alpha)

    print('Test GAT forward gpu kernels passed.')


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


def test_gat_backward_kernels():
    # Test GAT backward gpu kernels
    print('Test GAT backward gpu kernels...')
    rng = np.random.default_rng(42)
    feature_dim = 128
    A = utils.generate_sparse_matrix(1000, 1000, 50000, np.float32, rng)
    A.data[:] = 1.0
    dZ = utils.generate_dense_matrix(A.shape[0], feature_dim, np.float32, rng)
    Alpha_data = utils.generate_dense_matrix(A.data.shape[0], 1, np.float32,
                                             rng).squeeze()
    M = utils.generate_dense_matrix(A.shape[0], feature_dim, np.float32, rng)

    # Test gat_backward_dZ_dAlpha_rowdot
    dAlpha_data_ref = np.zeros(A.data.shape[0], dtype=np.float32)
    row_dot_ref = np.zeros(A.shape[0], dtype=np.float32)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            rowNo = i
            colNo = A.indices[j]
            dAlpha_data_ref[j] = np.dot(dZ[rowNo, :], M[colNo, :])
            row_dot_ref[rowNo] += dAlpha_data_ref[j] * Alpha_data[j]

    row_dot_dev = cp.zeros(A.shape[0], dtype=cp.float32)
    row_dot_dev_shfl = cp.zeros(A.shape[0], dtype=cp.float32)
    dAlpha_data_dev = cp.zeros_like(A.data, dtype=cp.float32)
    dAlpha_data_dev_shfl = cp.zeros_like(A.data, dtype=cp.float32)
    gat_backward_dZ_dAlpha_rowdot[min(65535,
                                      len(A.indptr) - 1),
                                  128](dAlpha_data_dev, row_dot_dev,
                                       cp.asarray(Alpha_data), cp.asarray(dZ),
                                       cp.asarray(M), cp.asarray(A.indices),
                                       cp.asarray(A.indptr))

    gat_backward_dZ_dAlpha_rowdot_shfl[min(65535,
                                           len(A.indptr) - 1),
                                       128](dAlpha_data_dev_shfl,
                                            row_dot_dev_shfl,
                                            cp.asarray(Alpha_data),
                                            cp.asarray(dZ), cp.asarray(M),
                                            cp.asarray(A.indices),
                                            cp.asarray(A.indptr))
    assert cp.allclose(dAlpha_data_dev, dAlpha_data_ref)
    assert cp.allclose(row_dot_dev, row_dot_ref)

    assert cp.allclose(dAlpha_data_dev_shfl, dAlpha_data_ref)
    assert cp.allclose(row_dot_dev_shfl, row_dot_ref)

    # Test gat_backward_softmax_leakyrelu_lr
    l = utils.generate_dense_matrix(A.shape[0], 1, np.float32, rng).squeeze()
    r = utils.generate_dense_matrix(A.shape[1], 1, np.float32, rng).squeeze()
    dE_data_ref = np.zeros(A.data.shape[0], dtype=np.float32)
    dl_ref = np.zeros(A.shape[0], dtype=np.float32)
    dr_ref = np.zeros(A.shape[1], dtype=np.float32)
    dl_dev = cp.zeros(A.shape[0], dtype=cp.float32)
    dr_dev = cp.zeros(A.shape[1], dtype=cp.float32)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            rowNo = i
            colNo = A.indices[j]
            dE_data_ref[j] = (dAlpha_data_ref[j] - np.dot(
                dAlpha_data_ref[A.indptr[i]:A.indptr[i + 1]],
                Alpha_data[A.indptr[i]:A.indptr[i + 1]])) * Alpha_data[j]
            D_recomp = l[rowNo] + r[colNo]
            if D_recomp <= 0:
                dE_data_ref[j] *= 0.2
            dl_ref[rowNo] += dE_data_ref[j]
            dr_ref[colNo] += dE_data_ref[j]
    gat_backward_softmax_leakyrelu_lr[min(65535,
                                          len(A.indptr) - 1),
                                      128](dl_dev, dr_dev, cp.asarray(l),
                                           cp.asarray(r),
                                           cp.asarray(Alpha_data),
                                           dAlpha_data_dev,
                                           cp.asarray(A.indices),
                                           cp.asarray(A.indptr), row_dot_dev)

    assert cp.allclose(dl_dev, dl_ref)
    assert cp.allclose(dr_dev, dr_ref)

    print('Test GAT backward gpu kernels passed.')


def benchmark_gat_backward_kernels(nodes=15,
                                   features=128,
                                   density=0.01,
                                   num_warmup=1,
                                   num_repeats=4):
    nodes = 2**nodes
    edges = density * nodes * nodes
    # Benchmark GAT backward gpu kernels
    print('Benchmark GAT backward gpu kernels...')
    rng = np.random.default_rng(42)
    A = utils.generate_sparse_matrix(nodes, nodes, edges, np.float32, rng)
    A.data[:] = 1.0
    dZ = utils.generate_dense_matrix(A.shape[0], features, np.float32, rng)
    Alpha_data = utils.generate_dense_matrix(A.data.shape[0], 1, np.float32,
                                             rng).squeeze()
    M = utils.generate_dense_matrix(A.shape[0], features, np.float32, rng)

    Alpha_data = cp.asarray(Alpha_data)
    dZ = cp.asarray(dZ)
    M = cp.asarray(M)
    indices = cp.asarray(A.indices)
    indptr = cp.asarray(A.indptr)

    # Benchmark gat_backward_dZ_dAlpha_rowdot
    for kernel in [
            gat_backward_dZ_dAlpha_rowdot, gat_backward_dZ_dAlpha_rowdot_shfl
    ]:
        row_dot_dev = cp.zeros(A.shape[0], dtype=cp.float32)
        dAlpha_data_dev = cp.zeros_like(A.data, dtype=cp.float32)
        

        gpu_setup = """row_dot_dev[:] = 0;dAlpha_data_dev[:] = 0;cp.cuda.get_current_stream().synchronize()"""

        gpu_stmt = """kernel[min(65535,len(A.indptr) - 1), 128](dAlpha_data_dev, row_dot_dev, Alpha_data, dZ, M, indices, indptr);cp.cuda.get_current_stream().synchronize()"""

        gpu_runtimes = repeat(gpu_stmt,
                              setup=gpu_setup,
                              repeat=num_warmup + num_repeats,
                              number=1,
                              globals={
                                  **locals(),
                                  **globals()
                              })
        print(
            f"{kernel.__name__} GPU: {np.median(gpu_runtimes[num_warmup:])} +- {np.std(gpu_runtimes[num_warmup:])}"
        )
    print('Benchmark GAT backward gpu kernels finished.')


if __name__ == "__main__":
    test_gat_forward_kernels()
    test_gat_backward_kernels()
    # benchmark_gat_backward_kernels()
