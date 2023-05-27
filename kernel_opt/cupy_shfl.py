import cupy as cp
import cupyx as cpx
from scipy import sparse

FULL_MASK = 0xFFFFFFFF


@cpx.jit.rawkernel()
def VA_f_0(out_data, indices, indptr, H_tile_1, H_tile_2):
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
def GAT_b_0(out_data, row_dot, Alpha_data, dZ, M, indices, indptr):
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
def AGNN_f_0(out_data, indices, indptr, H_tile_1, H_tile_2, H_tile_1_norm,
             H_tile_2_norm):
    """ Computes A * ((H x H^T) / (||H|| x ||H||^T)

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
            a = 0.0
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

    bid = cpx.jit.blockIdx.x  # Block ID
    num_blocks = cpx.jit.gridDim.x
    tid = cpx.jit.threadIdx.x  # Thread ID
    num_threads = cpx.jit.blockDim.x
    wid = tid // cpx.jit.warpsize  # Warp ID
    num_warps = cp.int32(cp.ceil(num_threads / cpx.jit.warpsize))
    twid = tid % cpx.jit.warpsize  # Thread ID within warp

    for i in range(bid, len(indptr) - 1, num_blocks):
        for j in range(indptr[i] + wid, indptr[i + 1], num_warps):
            rowNo = i
            colNo = indices[j]
            dQ = 0.0
            C = 0.0
            D = n_tile_1[rowNo] * n_tile_2[colNo]
            for k in range(twid, dZ.shape[1], cpx.jit.warpsize):
                dQ += dZ[rowNo, k] * M[colNo, k]
            for k in range(twid, H_tile_1.shape[1], cpx.jit.warpsize):
                C += H_tile_1[rowNo, k] * H_tile_2[colNo, k]

            dQ += cpx.jit.shfl_down_sync(FULL_MASK, dQ, 16)
            dQ += cpx.jit.shfl_down_sync(FULL_MASK, dQ, 8)
            dQ += cpx.jit.shfl_down_sync(FULL_MASK, dQ, 4)
            dQ += cpx.jit.shfl_down_sync(FULL_MASK, dQ, 2)
            dQ += cpx.jit.shfl_down_sync(FULL_MASK, dQ, 1)

            C += cpx.jit.shfl_down_sync(FULL_MASK, C, 16)
            C += cpx.jit.shfl_down_sync(FULL_MASK, C, 8)
            C += cpx.jit.shfl_down_sync(FULL_MASK, C, 4)
            C += cpx.jit.shfl_down_sync(FULL_MASK, C, 2)
            C += cpx.jit.shfl_down_sync(FULL_MASK, C, 1)

            if twid == 0:
                dC_out_data[j] = dQ / D
                dD_out_data[j] = -C * dQ / (D * D)