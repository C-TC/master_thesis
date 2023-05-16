import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import utils

from cupyx.profiler import benchmark


@cpx.jit.rawkernel()
def ahht(out_data, indices, indptr, H_tile_1, H_tile_2):

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


FULL_MASK = 0xFFFFFFFF


@cpx.jit.rawkernel()
def ahht_shfl(out_data, indices, indptr, H_tile_1, H_tile_2):

    """ Computes A * (H * H^T) 
    
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
            if twid == 0:
                out_data[j] = a


@cpx.jit.rawkernel()
def masked_dense(out_data, indices, indptr, Q_tile, D_tile , U_tile):

    """ Computes A * (Q @ D @ U^T), A is sparse (represented by indices and indptr), Q, D, U are dense

    TODO: still needs to be checked!

    Each block computes a subset of the rows of the output.
    """

    bid = cpx.jit.blockIdx.x  # Block ID
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x  # Thread ID
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            for k in range(Q_tile.shape[1]):
                for l in range(0, D_tile.shape[1]):
                        out_data[j] += Q_tile[rowNo, k] * D_tile[k, l] * U_tile[colNo, l]  # critical difference to ahht kernel is here: we expect U transposed, not N as input!


@cpx.jit.rawkernel()
def masked_dense_shfl(out_data, indices, indptr, Q_tile, D_tile , U_tile):

    """ Computes A * (Q @ D @ U^T), A is sparse (represented by indices and indptr), Q, D, U are dense

    TODO: still needs to be checked!

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
            for k in range(Q_tile.shape[1]):
                for l in range(twid, D_tile.shape[1], cpx.jit.warpsize):
            # for k in range(twid, Q_tile.shape[1], cpx.jit.warpsize):

            #     for l in range(0, D_tile.shape[1]):  # TODO: need other twid?

                    a += Q_tile[rowNo, k] * D_tile[k, l] * U_tile[colNo, l]  # critical difference to ahht kernel is here: we expect U transposed, not N as input!
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 16)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 8)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 4)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 2)
            a += cpx.jit.shfl_down_sync(FULL_MASK, a, 1)
            if twid == 0:
                out_data[j] = a


@cp.fuse()
def fuse_ahht(data, indices, indptr, H_tile_1, H_tile_2):
    out_tile = cp.zeros_like(data)
    for i in range(len(indptr) - 1):
        for j in range(indptr[i], indptr[i+1]):
            rowNo = i
            colNo = indices[j]
            for k in range(H_tile_1.shape[1]):
                out_tile[j] += H_tile_1[rowNo, k] * H_tile_2[colNo, k] # not transposed
            out_tile[j] *= data[j] 

    return out_tile


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
                out_data[j] += H_tile_1[rowNo, k] * H_tile_2[colNo, k]
            out_data[j] /= H_tile_1_norm[rowNo] * H_tile_2_norm[colNo]


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
            if twid == 0:
                out_data[j] = a / (H_tile_1_norm[rowNo] * H_tile_2_norm[colNo])


@cpx.jit.rawkernel()
def ac(out_data, indices, indptr, rowsum, a_LWH, HWa_H):
    """ Computes A * (1 x a_L^T x W^T x H^T + H x W x a_H x 1^T)
    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            out_data[j] = a_LWH[0, colNo] + HWa_H[rowNo, 0]
            cpx.jit.atomic_add(rowsum, rowNo, a_LWH[0, colNo] + HWa_H[rowNo, 0])


@cpx.jit.rawkernel()
def gat_C_D_forward(out_data, indices, indptr, l, r):
    """ Computes D = A * C
        where C(i, j) = l(i) + r(j)
        l,r are 1D array
    """

    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            out_data[j] = l[rowNo] + r[colNo]


@cpx.jit.rawkernel()
def gat_softmax_kernel_1(out_data, data, indices, indptr, rowmax):
    """Softmax on each row of E:
        part 1: exp(E(i,j) - rowmax(i))"""
    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            rowmax[rowNo] = max(rowmax[rowNo], data[j])
    
    cpx.jit.syncthreads()

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            out_data[j] = cp.exp(data[j] - rowmax[rowNo])


@cpx.jit.rawkernel()
def gat_softmax_kernel_2(out_data, indices, indptr, row_sum):
    """Softmax on each row of E:
        part 2: E(i,j) / row_sum(i)"""
    bid = cpx.jit.blockIdx.x
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x
    num_threads = cpx.jit.blockDim.x

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            cpx.jit.atomic_add(row_sum, rowNo, out_data[j])
    
    cpx.jit.syncthreads()

    for i in range(bid, (len(indptr) - 1), num_blocks):
        for j in range(indptr[i] + tid, indptr[i + 1], num_threads):
            rowNo = i
            colNo = indices[j]
            out_data[j] /= row_sum[rowNo]


def gat_softmax_rowise(data, indices, indptr):
    """Softmax on each row of E"""
    # TODO: fuse these two kernels into one
    tmp_data = cp.zeros_like(data)
    row_max = cp.zeros(len(indptr) - 1, dtype=data.dtype)
    gat_softmax_kernel_1[min(65535, len(indptr) - 1), 128](tmp_data, data, indices, indptr, row_max)
    row_sum = cp.zeros(len(indptr) - 1, dtype=data.dtype)
    gat_softmax_kernel_2[min(65535, len(indptr) - 1), 128](tmp_data, indices, indptr, row_sum)

    return tmp_data

def test_gat_softmax_rowise():
    """Test and benchmark softmax_rowise"""
    rng = np.random.default_rng(42)
    A = utils.generate_sparse_matrix(10000, 10000, 0.01, np.float32, rng)
    A.data[:] = 1
    E_data = utils.generate_dense_matrix(A.data.shape[0], 1, np.float32, rng).squeeze()
    E = sparse.csr_matrix((E_data, A.indices, A.indptr))
    E_dev = utils.sp2cp(E)

    rowmax = np.zeros(A.shape[0], dtype=np.float32)
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jidx]
            rowmax[i] = max(rowmax[i], E_data[jidx])
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jidx]
            E_data[jidx] = np.exp(E_data[jidx] - rowmax[i])
    
    rowsum = np.zeros(A.shape[0], dtype=np.float32)
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jidx]
            rowsum[i] += E_data[jidx]
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jidx]
            E_data[jidx] /= rowsum[i]
    ref_out = cp.asarray(E_data)
    out_dev = gat_softmax_rowise(E_dev.data, E_dev.indices, E_dev.indptr)
    assert cp.allclose(out_dev, ref_out)

    # Benchmark
    print("Benchmarking...")
    print(f"""GAT softmax rowise:
    {benchmark(
    gat_softmax_rowise,
    (E_dev.data, E_dev.indices, E_dev.indptr),
    n_warmup=10,
    n_repeat=100,
    name="GAT softmax rowise")}""")
    

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # Create a sparse matrix
    A = utils.generate_sparse_matrix(10000, 10000, 0.01, np.float32, rng)
    A.data = np.ones_like(A.data)
    # Create a dense matrix
    H = utils.generate_dense_matrix(10000, 128, np.float32, rng)
    H_dev = cp.asarray(H)

    val_0 = utils.sp2cp(A)
    val_0.data[:] = 0

    val_1 = utils.sp2cp(A)
    val_1.data[:] = 0

    ref = utils.sp2cp(A)
    HHT = H_dev @ H_dev.T
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = ref.indices[jidx]
            ref.data[jidx] = HHT[i, j]
    ref_data_1 = A.multiply(H @ H.T)

    assert cp.allclose(ref.data, ref_data_1.data)

    ahht[min(65535, A.shape[0]), 128](val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev)
    ahht_shfl[min(65535, A.shape[0]), 128](val_1.data, val_1.indices, val_1.indptr, H_dev, H_dev)

    assert cp.allclose(ref.data, val_0.data)
    print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)}")
    assert cp.allclose(ref.data, val_1.data)
    print(f"Relative error: {cp.linalg.norm(val_1.data - ref.data) / cp.linalg.norm(ref.data)}")

    # Benchmark
    # doesn't reset val_0.data to zero
    print("Benchmarking...")
    print(f"""AHHT simple:
{benchmark(
    ahht[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev),
    n_warmup=10,
    n_repeat=100)}""")
    print(f"""AHHT shuffle:
{benchmark(
    ahht_shfl[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev),
    n_warmup=10,
    n_repeat=100)}""")

    val_0.data[:] = 0
    val_1.data[:] = 0

    Q = utils.generate_dense_matrix(10000, 16, np.float32, rng)
    Q_dev = cp.asarray(Q)
    D = utils.generate_dense_matrix(16, 256, np.float32, rng)
    D_dev = cp.asarray(D)
    U = utils.generate_dense_matrix(10000, 256, np.float32, rng)
    U_dev = cp.asarray(U)
    
    QDUT = Q_dev @ D_dev @ U_dev.T
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = ref.indices[jidx]
            ref.data[jidx] = QDUT[i, j]
    ref_data_1 = A.multiply(Q @ D @ U.T)

    assert cp.allclose(ref.data, ref_data_1.data)

    masked_dense[min(65535, A.shape[0]), 128](val_0.data, val_0.indices, val_0.indptr, Q_dev, D_dev, U_dev)
    masked_dense_shfl[min(65535, A.shape[0]), 128](val_1.data, val_1.indices, val_1.indptr, Q_dev, D_dev, U_dev)

    assert cp.allclose(ref.data, val_0.data)
    print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)}")
    assert cp.allclose(ref.data, val_1.data)
    print(f"Relative error: {cp.linalg.norm(val_1.data - ref.data) / cp.linalg.norm(ref.data)}")

    Q_dev = None
    D_dev = None
    U_dev = None

    # Benchmark
    # doesn't reset val_0.data to zero
    print("Benchmarking...")
    print(f"""Masked dense simple:
{benchmark(
    ahht[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev),
    n_warmup=10,
    n_repeat=100)}""")
    print(f"""Masked dense shuffle:
{benchmark(
    ahht_shfl[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev),
    n_warmup=10,
    n_repeat=100)}""")

    val_0.data[:] = 0
    val_1.data[:] = 0

    H_norm = np.linalg.norm(H_dev, 2, axis = 1) # row-wise computation of the L2 norm
    H_norm_ref = H_norm.reshape(H_norm.shape[0], 1)
    HHT_norm = H_norm_ref @ H_norm_ref.T
    P = np.divide(HHT, HHT_norm)
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = ref.indices[jidx]
            ref.data[jidx] = P[i, j]

    H_norm_dev = cp.asarray(H_norm)

    ahhtnorm[min(65535, A.shape[0]), 128](val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev)
    ahhtnorm_shfl[min(65535, A.shape[0]), 128](val_1.data, val_1.indices, val_1.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev)

    assert cp.allclose(ref.data, val_0.data)
    print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)}")
    assert cp.allclose(ref.data, val_1.data)
    print(f"Relative error: {cp.linalg.norm(val_1.data - ref.data) / cp.linalg.norm(ref.data)}")

    # Benchmark
    # doesn't reset val_0.data to zero
    print("Benchmarking...")
    print(f"""AHHT Norm simple:
    {benchmark(
    ahhtnorm[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev),
    n_warmup=10,
    n_repeat=100)}""")
    print(f"""AHHT Norm shuffle:
    {benchmark(
    ahhtnorm_shfl[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, H_dev, H_dev, H_norm_dev, H_norm_dev),
    n_warmup=10,
    n_repeat=100)}""")

    val_0.data[:] = 0

    # Create a dense matrix
    W = utils.generate_dense_matrix(128, 128, np.float32, rng)
    W_dev = cp.asarray(W)
    a_L = utils.generate_dense_matrix(1, 128, np.float32, rng) # already transposed
    a_L_dev = cp.asarray(a_L)
    a_H = utils.generate_dense_matrix(128, 1, np.float32, rng)
    a_H_dev = cp.asarray(a_H)

    one = np.ones((H_dev.shape[0], 1), H_dev.dtype)
    one_dev = cp.asarray(one)

    a_LWH = a_L_dev @ W_dev.T @ H_dev.T
    HWa_H = H_dev @ W_dev @ a_H_dev
    C = one_dev @ a_LWH + HWa_H @ one_dev.T
    ref_rowsum = np.zeros(H_dev.shape[0], H_dev.dtype)
    ref_rowsum_dev = cp.asarray(ref_rowsum)
    for i in range(A.shape[0]):
        for jidx in range(A.indptr[i], A.indptr[i+1]):
            j = ref.indices[jidx]
            ref.data[jidx] = C[i, j]
            ref_rowsum_dev[i] += C[i, j]

    rowsum = np.zeros(H_dev.shape[0], H_dev.dtype)
    rowsum_dev = cp.asarray(rowsum)

    ac[min(65535, A.shape[0]), 128](val_0.data, val_0.indices, val_0.indptr, rowsum_dev, a_LWH, HWa_H)

    assert cp.allclose(ref.data, val_0.data)
    assert cp.allclose(ref_rowsum_dev, rowsum_dev)
    print(f"Relative error: {cp.linalg.norm(val_0.data - ref.data) / cp.linalg.norm(ref.data)} {cp.linalg.norm(rowsum_dev - ref_rowsum_dev) / cp.linalg.norm(ref_rowsum_dev)}")

    # Benchmark
    print("Benchmarking...")
    # doesn't reset rowsum_dev to zero
    print(f"""AC simple:
    {benchmark(
    ac[min(65535, A.shape[0]), 128],
    (val_0.data, val_0.indices, val_0.indptr, rowsum_dev, a_LWH, HWa_H),
    n_warmup=10,
    n_repeat=100)}""")

    test_gat_softmax_rowise()
