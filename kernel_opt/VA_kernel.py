import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import dace


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


M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz = dace.symbol('nnz')

@dace.program
def ahht_dace(out_data: dace.float32[nnz], vals:dace.float32[nnz], indices: dace.int32[nnz], indptr: dace.int32[M + 1], H_tile_1: dace.float32[M, K], H_tile_2: dace.float32[N, K]):
    for i in dace.map[M]:
        for j in dace.map[indptr[i]: indptr[(i+1)]]:
            rowNo = i
            colNo = indices[j]
            for k in dace.map[K]:
                out_data[j] += vals[j] * H_tile_1[rowNo, k] * H_tile_2[colNo, k]
