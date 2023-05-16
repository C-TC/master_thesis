import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import utils

from cupyx.profiler import benchmark

FULL_MASK = 0xFFFFFFFF


@cpx.jit.rawkernel()
def had_div(out_data_1, out_data_2, indices, indptr,
            dZ_tile_1, H_tile_1, H_tile_2, W_T, n_tile_1, n_tile_2):

    """ Computes both
    1. (H @ H^T) * (A * (dZ @ W^T @ H^T) / (n_tile_1 @ n_tile_1^T)^2 in output1 and
    2. A * (dZ @ W^T @ H^T) / ((H @ H^T) * (A * (dZ @ W^T @ H^T) / (n_tile_1 @ n_tile_1^T)^2) in output2

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
            tmp1 = (n_tile_1[rowNo, 0] * n_tile_2[colNo, 0])
            minus_C = 0.0
            dP = 0.0
            for k in range(H_tile_1.shape[1]):
                minus_C += (-1) * H_tile_1[rowNo, k] * H_tile_2[colNo, k]
                for l in range(W_T.shape[1]):
                    dP += dZ_tile_1[rowNo, k] * W_T[k, l] * H_tile_2[colNo, l] # dP
            out_data_1[j] = minus_C * dP / (tmp1**2)
            out_data_2[j] = dP / out_data_1[j]

@cpx.jit.rawkernel()
def ahhtnorm(out_data, indices, indptr, H_tile_1, H_tile_2, H_tile_1_norm, H_tile_2_norm):

    """ Computes A * ((H x H^T) / (||H|| x ||H||^T)

    H^T is done inside the kernel by switching the indices.

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
                out_data[j] += (H_tile_1[rowNo, k] * H_tile_2[colNo, k])
            out_data[j] /= (H_tile_1_norm[rowNo, 0] * H_tile_2_norm[colNo, 0])

@cpx.jit.rawkernel()
def experimental_norm(out_data, indices, indptr, H_tile_1, H_tile_2, H_tile_1_norm, H_tile_2_norm):

    """ Computes A * ((H x H^T) / (||H|| x ||H||^T)

    H^T is done inside the kernel by switching the indices.

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
            out_data[j] = (H_tile_1_norm[rowNo, 0] * H_tile_2_norm[colNo, 0])


@cpx.jit.rawkernel()
def ahhtnorm_shfl(out_data, indices, indptr, H_tile_1, H_tile_2, H_tile_1_norm, H_tile_2_norm):

    """ Computes A * ((H * H^T) / (||H|| x ||H||^T)

    H^T is done inside the kernel by switching the indices.

    Warp shuffle version.
    Each block computes a subset of the rows of the output. Each warp computes a subset of the columns of the parent
    block's rows. Using grid-strided pattern.
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
            a = cp.float32(0)
            for k in range(twid, H_tile_1.shape[1], cpx.jit.warpsize):
                a += H_tile_1[rowNo, k] * H_tile_2[colNo, k]
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 16)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 8)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 4)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 2)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 1)

            tmp = (H_tile_1_norm[rowNo, 0] * H_tile_2_norm[colNo, 0])
            if (twid == 0) and tmp!=0:
                out_data[j] = a / tmp

            # out_tile = cp.array(tmp.multiply(cp.asnumpy(np.divide(H_tile_1 @ H_tile_2.T, H_tile_1_norm_gpu @ H_tile_2_norm_gpu.T))) @ (H[k:min(k + tau, H.shape[0]), :] @ W))


if __name__ == "__main__":

    print("No tests impelemented yet.")
#     rng = np.random.default_rng(42)
#
#     # Create a sparse matrix
#     A = utils.generate_sparse_matrix(10000, 10000, 0.01, np.float32, rng)
#     # Create a dense matrix
#     H = utils.generate_dense_matrix(10000, 128, np.float32, rng)
#     H_dev = cp.asarray(H)
#
#     val_0 = utils.sp2cp(A)
#     val_0.data[:] = 0
#
#     val_1 = utils.sp2cp(A)
#     val_1.data[:] = 0
#
#     ref = utils.sp2cp(A)
#     HHT = H_dev @ H_dev.T
#     for i in range(A.shape[0]):
#         for jidx in range(A.indptr[i], A.indptr[i+1]):
#             j = ref.indices[jidx]
#             ref.data[jidx] = HHT[i, j]
#
#     assert cp.allclose(ref.data, val_0.data)
#     print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)}")
#     assert cp.allclose(ref.data, val_1.data)
#     print(f"Relative error: {cp.linalg.norm(val_1.data - ref.data) / cp.linalg.norm(ref.data)}")
#
#     # Benchmark
#     # doesn't reset val_0.data to zero
#     print("Benchmarking...")
#     print(f"""AHHT simple:
# {benchmark(
#     ahht[min(65535, A.shape[0]), 128],
#     (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev),
#     n_warmup=10,
#     n_repeat=100)}""")
#     print(f"""AHHT shuffle:
# {benchmark(
#     ahht_shfl[min(65535, A.shape[0]), 128],
#     (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev),
#     n_warmup=10,
#     n_repeat=100)}""")
#
#     val_0.data[:] = 0
#     val_1.data[:] = 0
#
#     H_norm = np.linalg.norm(H_dev, 2, axis = 1) # row-wise computation of the L2 norm
#     H_norm_ref = H_norm.reshape(H_norm.shape[0], 1)
#     HHT_norm = H_norm_ref @ H_norm_ref.T
#     P = np.divide(HHT, HHT_norm)
#     for i in range(A.shape[0]):
#         for jidx in range(A.indptr[i], A.indptr[i+1]):
#             j = ref.indices[jidx]
#             ref.data[jidx] = P[i, j]
#
#     H_norm_dev = cp.asarray(H_norm)
#
#     ahhtnorm[min(65535, A.shape[0]), 128](val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev)
#     ahhtnorm_shfl[min(65535, A.shape[0]), 128](val_1.data, val_1.indices, val_1.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev)
#
#     assert cp.allclose(ref.data, val_0.data)
#     print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)}")
#     assert cp.allclose(ref.data, val_1.data)
#     print(f"Relative error: {cp.linalg.norm(val_1.data - ref.data) / cp.linalg.norm(ref.data)}")
#
#     # Benchmark
#     # doesn't reset val_0.data to zero
#     print("Benchmarking...")
#     print(f"""AHHT Norm simple:
#     {benchmark(
#     ahhtnorm[min(65535, A.shape[0]), 128],
#     (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev),
#     n_warmup=10,
#     n_repeat=100)}""")
#     print(f"""AHHT Norm shuffle:
#     {benchmark(
#     ahhtnorm_shfl[min(65535, A.shape[0]), 128],
#     (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev),
#     n_warmup=10,
#     n_repeat=100)}""")
#
#     val_0.data[:] = 0
#
#     # Create a dense matrix
#     W = utils.generate_dense_matrix(128, 128, np.float32, rng)
#     W_dev = cp.asarray(W)
#     a_L = utils.generate_dense_matrix(1, 128, np.float32, rng) # already transposed
#     a_L_dev = cp.asarray(a_L)
#     a_H = utils.generate_dense_matrix(128, 1, np.float32, rng)
#     a_H_dev = cp.asarray(a_H)
#
#     one = np.ones((H_dev.shape[0], 1), H_dev.dtype)
#     one_dev = cp.asarray(one)
#
#     a_LWH = a_L_dev @ W_dev.T @ H_dev.T
#     HWa_H = H_dev @ W_dev @ a_H_dev
#     C = one_dev @ a_LWH + HWa_H @ one_dev.T
#     ref_rowsum = np.zeros(H_dev.shape[0], H_dev.dtype)
#     ref_rowsum_dev = cp.asarray(ref_rowsum)
#     for i in range(A.shape[0]):
#         for jidx in range(A.indptr[i], A.indptr[i+1]):
#             j = ref.indices[jidx]
#             ref.data[jidx] = C[i, j]
#             ref_rowsum_dev[i] += C[i, j]
#
#     rowsum = np.zeros(H_dev.shape[0], H_dev.dtype)
#     rowsum_dev = cp.asarray(rowsum)
#
#     ac[min(65535, A.shape[0]), 128](val_0.data, val_0.indices, val_0.indptr, rowsum_dev, a_LWH, HWa_H)
#
#     assert cp.allclose(ref.data, val_0.data)
#     assert cp.allclose(ref_rowsum_dev, rowsum_dev)
#     print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)} {cp.linalg.norm(rowsum_dev - ref_rowsum_dev) / cp.linalg.norm(ref_rowsum_dev)}")
#
#     # Benchmark
#     print("Benchmarking...")
#     # doesn't reset rowsum_dev to zero
#     print(f"""AC simple:
#     {benchmark(
#     ac[min(65535, A.shape[0]), 128],
#     (val_0.data, val_0.indices, val_0.indptr, rowsum_dev, a_LWH, HWa_H),
#     n_warmup=10,
#     n_repeat=100)}""")
#
#     test_gat_softmax_rowise()
