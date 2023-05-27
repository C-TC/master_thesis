import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import dace

FULL_MASK = 0xFFFFFFFF

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz = dace.symbol('nnz')


@cpx.jit.rawkernel()
def forward_ahhtnorm(out_data, indices, indptr, H_tile_1, H_tile_2,
                     H_tile_1_norm, H_tile_2_norm):
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
def forward_ahhtnorm_shfl(out_data, indices, indptr, H_tile_1, H_tile_2,
                          H_tile_1_norm, H_tile_2_norm):
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

@dace.program
def ahhtnorm_dace(out_data: dace.float32[nnz], indices: dace.float32[nnz], indptr: dace.float32[M+1], H_tile_1: dace.float32[M,K], H_tile_2: dace.float32[N,K], H_tile_1_norm: dace.float32[M], H_tile_2_norm: dace.float32[N]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            colNo = indices[j]
            tmp = 0.0
            for k in dace.map[0:K]:
                tmp += (H_tile_1[rowNo, k] * H_tile_2[colNo, k])
            out_data[j] = tmp / (H_tile_1_norm[rowNo] * H_tile_2_norm[colNo])




@cpx.jit.rawkernel()
def backward_Z_Q_CD(dC_out_data, dD_out_data, indices, indptr, dZ, M, H_tile_1,
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


@cpx.jit.rawkernel()
def backward_Z_Q_CD_shfl(dC_out_data, dD_out_data, indices, indptr, dZ, M, H_tile_1,
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


@dace.program
def Z_Q_CD_dace(dC_out_data: dace.float32[nnz], dD_out_data: dace.float32[nnz], indices: dace.float32[nnz], indptr: dace.float32[M+1], dZ: dace.float32[M,K], M_mat: dace.float32[N,K], H_tile_1: dace.float32[M,K], H_tile_2: dace.float32[N,K], n_tile_1: dace.float32[M], n_tile_2: dace.float32[N]):
    dC_out_data[:] = 0.0
    dD_out_data[:] = 0.0
    
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]: indptr[i+1]]:
            rowNo = i
            colNo = indices[j]
            dQ = 0.0
            C = 0.0
            D = n_tile_1[rowNo] * n_tile_2[colNo]
            for k in dace.map[0:K]:
                dQ += dZ[rowNo, k] * M_mat[colNo, k]
            for k in dace.map[0:K]:
                C += H_tile_1[rowNo, k] * H_tile_2[colNo, k]
            
            dC_out_data[j] = dQ / D
            dD_out_data[j] = -C * dQ / (D * D)

sdfg = Z_Q_CD_dace.to_sdfg(simplify=True)
sdfg.view()