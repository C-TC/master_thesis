import cupy as cp
import cupyx as cpx
import numpy as np
from scipy import sparse
import utils

from cupyx.profiler import benchmark

@cpx.jit.rawkernel()
def forward_ahhtnorm(out_data, indices, indptr, H_tile_1, H_tile_2, H_tile_1_norm, H_tile_2_norm):

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

def test_agnn_forward_kernels():
    
    # Test AGNN forward gpu kernels
    print('Test AGNN forward gpu kernels...')
    rng = np.random.default_rng(42)
    A = utils.generate_sparse_matrix(1000, 1000, 50000, np.float32, rng)
    A.data[:] = 1.0
    H = utils.generate_dense_matrix(1000, 128, np.float32, rng)
    H_norm = np.linalg.norm(H, axis=1, ord=2)

    # Test forward_ahhtnorm
    ref_Q_data = np.zeros_like(A.data)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            rowNo = i
            colNo = A.indices[j]
            tmp = H[rowNo, :] @ H[colNo, :].T
            ref_Q_data[j] = tmp / (H_norm[rowNo] * H_norm[colNo])
    Q_data = cp.zeros_like(A.data)
    forward_ahhtnorm[min(65535, len(A.indptr) - 1), 128](Q_data, cp.asarray(A.indices), cp.asarray(A.indptr), cp.asarray(H), cp.asarray(H), cp.asarray(H_norm), cp.asarray(H_norm))
    
    assert cp.allclose(Q_data, ref_Q_data)
    print('Test AGNN forward gpu kernels passed.')


@cpx.jit.rawkernel()
def backward_Z_Q_CD(dC_out_data, dD_out_data, indices, indptr, dZ, M, H_tile_1, H_tile_2, n_tile_1, n_tile_2):

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

def test_agnn_backward_kernels():
    # Test AGNN forward gpu kernels
    print('Test AGNN backward gpu kernels...')
    rng = np.random.default_rng(42)
    A = utils.generate_sparse_matrix(1000, 1000, 50000, np.float32, rng)
    A.data[:] = 1.0
    H = utils.generate_dense_matrix(1000, 128, np.float32, rng)
    H_norm = np.linalg.norm(H, axis=1, ord=2)
    dZ = utils.generate_dense_matrix(1000, 128, np.float32, rng)
    M = utils.generate_dense_matrix(1000, 128, np.float32, rng)

    # Test backward_Z_Q_CD
    ref_dC_data = np.zeros_like(A.data)
    ref_dD_data = np.zeros_like(A.data)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            rowNo = i
            colNo = A.indices[j]
            dQ_data_k = dZ[rowNo, :] @ M[colNo, :].T
            C = H[rowNo, :] @ H[colNo, :].T
            D = H_norm[rowNo] * H_norm[colNo]
            ref_dD_data[j] = - dQ_data_k * C / D**2
            ref_dC_data[j] = dQ_data_k / D
    dC_data = cp.zeros_like(A.data)
    dD_data = cp.zeros_like(A.data)
    backward_Z_Q_CD[min(65535, len(A.indptr) - 1), 128](dC_data, dD_data, cp.asarray(A.indices), cp.asarray(A.indptr), cp.asarray(dZ), cp.asarray(M), cp.asarray(H), cp.asarray(H), cp.asarray(H_norm), cp.asarray(H_norm))
    
    assert cp.allclose(dC_data, ref_dC_data)
    assert cp.allclose(dD_data, ref_dD_data)
    print('Test AGNN forward gpu kernels passed.')



if __name__ == "__main__":
    test_agnn_forward_kernels()
    test_agnn_backward_kernels()
