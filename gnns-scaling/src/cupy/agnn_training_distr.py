import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import kernels
import kernels_agnn
import utils

from mpi4py import MPI
from scipy import sparse
from timeit import repeat
from typing import List, Tuple, Union


def generate_blocks_inference(A: sparse.csr_matrix, H0_cols: int,
                              W_cols: List[int]) -> Tuple[int, List[sparse.csr_matrix]]:
    """ Generates the blocks for inference.

    The method splits the computation to batches by tiling the adjaceney matrix in blocks of size T x T. The tile size T is
    computed based on the available GPU memory

    :param A: The adjacency matrix.
    :param H0_cols: The number of columns in the initial H matrix.
    :param W_cols: The number of columns in the W matrices.
    :return: The tile size T and the blocks of A.
    """
    # A: NI x NK (sparse)
    # H: NK x NJ (dense)
    # W: NJ x NL (dense)
    # For simplicity, we assume that max(NJ) == max(NL) == max_cols, and that max_cols << NI, NK
    # A: NI x NK (sparse)
    # H: NK x max_cols (dense)
    # W: max_cols x max_cols (dense)
    # We perform the computation in blocks by tiling each of {NI, NK} in T tiles.

    # We need the following blocks for the forward pass:
    # W: max_cols x max_cols (dense)
    # A: T x T (sparse)
    # H_1: T x max_cols (dense)
    # H_2: T x max_cols (dense)
    # ||H_1||: T x 1 (dense)
    # ||H_2||: T x 1 (dense)
    # H_2 @ W: T x max_cols (dense)
    # Z: T x max_cols (dense)
    #
    # We require the following memory:
    # W: dtype*max_cols*max_cols
    # A: density*dtype*T^2 (data) + density*4*T^2 (indices) + 4*T (indptr)
    # H_1: dtype*T*max_cols
    # H_2: dtype*T*max_cols
    # ||H_1||: dtype*T
    # ||H_2||: dtype*T
    # H_2 @ W: dtype*T*max_cols
    # Z: dtype*T*max_cols
    # Total: T^2*(density*dtype + density*4) + T*(4 + 2*dtype + 4*dtype*max_cols) max_cols^2*dtype

    # We need the following blocks for the backward pass:
    # W: max_cols x max_cols (dense)
    # H_1: T x max_cols (dense)
    # H_2: T x max_cols (dense)
    # ||H_1||: T x 1 (dense)
    # ||H_2||: T x 1 (dense)
    # dCH: T x max_cols (dense)
    # dDn: T x max_cols (dense)
    # dH: T x max_cols (dense)
    # Z: T x max_cols (dense)
    # dZ: T x max_cols (dense)
    # A_THHTnnT: T x T (sparse)
    # QT: T x T (sparse)
    # dM: T x max_cols (dense)
    # dP_div_dD: T x T (sparse)
    # C_dP_div_D_sq: T x T (sparse)
    # dP_T_div_dD_T: T x T (sparse)
    # C_dP_div_D_sq_T: T x T (sparse)
    # Total: T^2*(5*density*dtype + density*20) + T*(20 + 2*dtype + 8*dtype*max_cols) + max_cols^2*dtype

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    max_cols = max(H0_cols, max(W_cols))
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]  # * 0.10  # just for local testing

    alpha = 5 * density * dtype + density * 20
    beta = 20 + 2 * dtype + 8 * dtype * max_cols
    gamma = max_cols**2 * dtype - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau = int(np.ceil((-beta + abs(delta)) / (2 * alpha)))

    num_tiles_i = int(np.ceil(A.shape[0] / tau))
    num_tiles_k = int(np.ceil(A.shape[1] / tau))
    print(f"Using {num_tiles_i}x{num_tiles_k} batches for inference.")

    A_blocks = []
    if tau >= A.shape[0] and tau >= A.shape[1]:
        A_blocks.append(A)
    else:
        for i in range(0, A.shape[0], tau):
            for k in range(0, A.shape[1], tau):
                tmp = sparse.csr_matrix(A[i:min(i + tau, A.shape[0]), k:min(k + tau, A.shape[1])])
                tmp.sum_duplicates()
                tmp.sort_indices()
                A_blocks.append(tmp)

    return tau, A_blocks


def cpu_computation(A: sparse.csr_matrix, H_tile_1: np.ndarray, H_tile_2: np.ndarray, W: List[np.ndarray],
                    bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm, layers: int, bcast_root: int,
                    reduce_root: int, expected_output_tile: np.ndarray, alpha: float, grad_of_loss) -> List[np.ndarray]:
    """ Performs computations of all layers on the CPU.

    :param A: The adjacency matrix tile.
    :param H_tile_1, H_tile_2: Tiles of the H matrix.
    :param W: The W matrices.
    :param bcast_comm: The bcast communicator.
    :param reduce_comm: The reduce communicator.
    :param layers: The number of layers for AGNN inference.
    :param bcast_root: The column of the process grid.
    :param reduce_root: The row of the process grid.
    :param expected_output_tile: The tile of the expected output.
    :param alpha: ???
    :param grad_of_loss: loss function
    :return: List of updated W.
    """

    H0_tile_x = H_tile_2
    H0_tile_y = H_tile_1
    H_tile_x = H_tile_2.copy()
    H_tile_y = H_tile_1.copy()

    Z_tile_x_list = []
    Z_tile_y_list = []

    W_new = []
    for i in range(len(W)):
        W_new.append(W[i].copy())

    bcast_rank = bcast_comm.Get_rank()
    reduce_rank = reduce_comm.Get_rank()

    # forward pass
    for i in range(layers):
        H_tile_x_norm = np.linalg.norm(H_tile_x, 2, axis = 1)
        H_tile_x_norm = H_tile_x_norm.reshape(H_tile_x_norm.shape[0], 1)
        H_tile_y_norm = np.linalg.norm(H_tile_y, 2, axis = 1)
        H_tile_y_norm = H_tile_y_norm.reshape(H_tile_y_norm.shape[0], 1)
        Z_tile = A.multiply(np.divide(H_tile_y @ H_tile_x.T, H_tile_y_norm @ H_tile_x_norm.T)) @ (H_tile_x @ W[i])
        reduce_comm.Allreduce(MPI.IN_PLACE, Z_tile, op=MPI.SUM)
        H_tile_y = np.maximum(Z_tile, 0)
        Z_tile_y_list.append(Z_tile.copy())
        utils.bcast_matrix(Z_tile, bcast_comm, bcast_root)
        if i < layers - 1:
            Z_tile_x_list.append(Z_tile.copy())
            H_tile_x = np.maximum(Z_tile, 0)

    # Z_tile_y_list has layers elements and is replicated across the y dimension
    # Z_tile_x_list has layers-1 elements and is replicated across the x dimension

    if grad_of_loss is None:
        dH_tile = H_tile_y - expected_output_tile
    else:
        dH_tile = grad_of_loss(expected_output_tile, H_tile_y)

    # backward pass
    for i in range(layers - 1, -1, -1):
        if i > 0:
            H_tile_x = np.maximum(Z_tile_x_list[i-1], 0)
            H_tile_y = np.maximum(Z_tile_y_list[i-1], 0)
        else:
            H_tile_x = H0_tile_x
            H_tile_y = H0_tile_y

        # dZ is replicated across the x dimension
        dZ_tile = np.multiply((Z_tile_y_list[i] > 0), dH_tile)

        H_tile_x_norm = np.linalg.norm(H_tile_x, 2, axis = 1)
        H_tile_x_norm = H_tile_x_norm.reshape(H_tile_x_norm.shape[0], 1)
        H_tile_y_norm = np.linalg.norm(H_tile_y, 2, axis = 1)
        H_tile_y_norm = H_tile_y_norm.reshape(H_tile_y_norm.shape[0], 1)

        Q = A.multiply(np.divide(H_tile_y @ H_tile_x.T, H_tile_y_norm @ H_tile_x_norm.T))

        dM_tile = Q.T @ dZ_tile # dM_tile is replicated across the y dimension
        bcast_comm.Allreduce(MPI.IN_PLACE, dM_tile, op=MPI.SUM)

        if i > 0:
            # dP is distributed like A
            dP_tile = A.multiply(dZ_tile @ W[i].T @ H_tile_x.T) # Is this correct? Because of M^T
            # dD is distributed like A
            dD_tile = -np.multiply(H_tile_y @ H_tile_x.T, dP_tile /((H_tile_y_norm @ H_tile_x_norm.T) ** 2))
            # dC is distributed like A
            # NOTE: Should division by zero happen?
            dC_tile = np.nan_to_num( dP_tile / dD_tile ) # converts nan to 0

            # dn_tile is only significant on the process on the diagonal
            dn_tile = dD_tile @ H_tile_x_norm
            if reduce_rank == reduce_root:
                reduce_comm.Reduce(MPI.IN_PLACE, dn_tile, op=MPI.SUM, root=reduce_root)
            else:
                reduce_comm.Reduce(dn_tile, None, op=MPI.SUM, root=reduce_root)
            tmp = dD_tile.T @ H_tile_y_norm
            if bcast_rank == bcast_root:
                bcast_comm.Reduce(MPI.IN_PLACE, tmp, op=MPI.SUM, root=bcast_root)
                dn_tile += tmp
            else:
                bcast_comm.Reduce(tmp, None, op=MPI.SUM, root=bcast_root)

            # compute dH
            dH_tile = dC_tile.T @ H_tile_y
            if bcast_rank == bcast_root and reduce_rank == reduce_root:
                 dH_tile += np.multiply(H_tile_y, (dn_tile / H_tile_y_norm)) + dM_tile @ W[i].T
            if bcast_rank == bcast_root:
                bcast_comm.Reduce(MPI.IN_PLACE, dH_tile, op=MPI.SUM, root=bcast_root)
            else:
                bcast_comm.Reduce(dH_tile, None, op=MPI.SUM, root=bcast_root)

            if reduce_rank == reduce_root:
                dH_tile += dC_tile @ H_tile_x
            else:
                dH_tile = dC_tile @ H_tile_x
            reduce_comm.Allreduce(MPI.IN_PLACE, dH_tile, op=MPI.SUM)

            dW = H_tile_x.T @ dM_tile
        else:
            dW = H0_tile_x.T @ dM_tile
        reduce_comm.Allreduce(MPI.IN_PLACE, dW, op=MPI.SUM)

        W_new[i] -= alpha * dW

    return W_new


def gpu_computation(A: sparse.csr_matrix, A_blocks: List[sparse.csr_matrix], H_tile_1: np.ndarray,
                    H_tile_2: np.ndarray, W: List[np.ndarray], tau: int, bcast_comm: MPI.Cartcomm,
                    reduce_comm: MPI.Cartcomm, layers: int, bcast_root: int, reduce_root: int,
                    expected_output_tile: np.ndarray, alpha: float, grad_of_loss) -> np.ndarray:
    """ Performs computations of all layers on the GPU
    :param A: The adjacency matrix tile.
    :param A_blocks: The blocks of A.
    :param H_tile_1, H_tile_2: Tiles of the H matrix.
    :param W: The W matrices.
    :param tau: The tile size.
    :param bcast_comm: The bcast communicator.
    :param reduce_comm: The reduce communicator.
    :param layers: The number of layers for AGNN inference.
    :param bcast_root: The row of the process grid.
    :param reduce_root: The row of the process grid.
    :param expected_output_tile: The tile of the expected output.
    :param alpha: ???
    :param grad_of_loss: loss function
    :return: The output (new H matrix tile after the last layer).
    """

    H0_tile_x = H_tile_2
    H0_tile_y = H_tile_1
    H_tile_x = H_tile_2.copy()
    H_tile_y = H_tile_1.copy()

    Z_tile_x_list = []
    Z_tile_y_list = []

    W_new = []
    for i in range(len(W)):
        W_new.append(W[i].copy())

    bcast_rank = bcast_comm.Get_rank()
    reduce_rank = reduce_comm.Get_rank()

    # forward pass
    for l in range(layers):
        W_gpu = cp.asarray(W[l])
        block_idx = 0
        Z_tile = np.zeros((A.shape[0], W[l].shape[1]), dtype=W[l].dtype)
        for i in range(0, A.shape[0], tau):
            Z_tile_tile = cp.asarray(Z_tile[i:min(i + tau, A.shape[0]), :])
            H_tile_y_tile = cp.asarray(H_tile_y[i:min(i + tau, H_tile_y.shape[0]), :])
            H_tile_y_tile_norm = np.expand_dims(np.linalg.norm(H_tile_y_tile, 2, axis = 1), axis=1) # row-wise computation of the L2 norm
            H_tile_y_tile_norm_gpu = cp.asarray(H_tile_y_tile_norm)
            for k in range(0, H_tile_x.shape[0], tau):
                tmp = A_blocks[block_idx]
                AHHTnorm = utils.sp2cp(tmp)
                AHHTnorm.data[:] = 0
                H_tile_x_tile = cp.asarray(H_tile_x[k:min(k + tau, H_tile_x.shape[0]), :])
                H_tile_x_tile_norm = np.expand_dims(np.linalg.norm(H_tile_x_tile, 2, axis = 1), axis=1) # row-wise computation of the L2 norm
                H_tile_x_tile_norm_gpu = cp.asarray(H_tile_x_tile_norm)
                kernels_agnn.ahhtnorm_shfl[min(65535, tmp.shape[0]), 128](AHHTnorm.data, AHHTnorm.indices, AHHTnorm.indptr,
                    H_tile_y_tile, H_tile_x_tile, H_tile_y_tile_norm_gpu, H_tile_x_tile_norm_gpu)
                Z_tile_tile += AHHTnorm @ (H_tile_x_tile @ W_gpu)
                block_idx += 1
            # NOTE: Assuming here a non-CUDA-aware MPI implementation.
            Z_tile[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(Z_tile_tile)
        reduce_comm.Allreduce(MPI.IN_PLACE, Z_tile, op=MPI.SUM)
        Z_tile_y_list.append(Z_tile.copy())
        H_tile_y = np.maximum(Z_tile, 0)
        utils.bcast_matrix(Z_tile, bcast_comm, bcast_root)
        if l < layers - 1:
            Z_tile_x_list.append(Z_tile.copy())
            H_tile_x = np.maximum(Z_tile, 0)

    # Z_tile_y_list has layers elements and is replicated across the y dimension
    # Z_tile_x_list has layers-1 elements and is replicated across the x dimension

    if grad_of_loss is None:
        dH_tile = H_tile_y - expected_output_tile
    else:
        dH_tile = grad_of_loss(expected_output_tile, H_tile_y)

    # backward pass
    for l in range(layers - 1, -1, -1):
        W_T_gpu = cp.asarray(W_new[l].T)

        if l > 0:
            dCH_tile = np.zeros(shape=(A.shape[0], Z_tile_x_list[l-1].shape[1]), dtype=A.dtype) # =(dC)@H
            dC_TH_tile = np.zeros(shape=(A.shape[0], Z_tile_y_list[l-1].shape[1]), dtype=A.dtype) # =(dCT)@H
            dDn_tile = np.zeros(shape=(A.shape[0], 1), dtype=A.dtype)
            dD_Tn_tile = np.zeros(shape=(A.shape[0], 1), dtype=A.dtype)

        dW = np.zeros(shape=W_new[l].shape, dtype=A.dtype)
        dM_tile = np.zeros(shape=(A.shape[0], W_new[l].shape[1]), dtype=A.dtype)

        block_idx = 0

        for i in range(0, A.shape[0], tau):
            if l > 0:
                H_tile_y_tile = cp.asarray(np.maximum(Z_tile_y_list[l-1][i:min(i + tau, A.shape[0]), :], 0))
                dCH_tile_tile = cp.asarray(dCH_tile[i:min(i + tau, A.shape[0]), :])
                dDn_tile_tile = cp.asarray(dDn_tile[i:min(i + tau, A.shape[0]), :])
            else:
                H_tile_y_tile = cp.asarray(H0_tile_y[i:min(i + tau, A.shape[0]), :])
            H_tile_y_tile_norm = cp.asarray(np.expand_dims(np.linalg.norm(H_tile_y_tile, 2, axis = 1), axis=1)) # row-wise computation of the L2

            dH_tile_tile = cp.asarray(dH_tile[i:min(i + tau, A.shape[0]), :])
            Z_tile_y_tile = cp.asarray(Z_tile_y_list[l][i:min(i + tau, A.shape[0]), :])
            dZ_tile_tile = cp.multiply((Z_tile_y_tile > 0), dH_tile_tile)

            # inner loop: horizontal tiling of A
            for k in range(0, H_tile_x.shape[0], tau):
                A_tile = A_blocks[block_idx]

                block_idx += 1

                if l > 0:
                    H_tile_x_tile = cp.asarray(np.maximum(Z_tile_x_list[l-1][k:min(k + tau, A.shape[0]), :], 0))
                else:
                    H_tile_x_tile = cp.asarray(H0_tile_x[k:min(k + tau, A.shape[0]), :])
                H_tile_x_tile_norm = cp.asarray(np.expand_dims(np.linalg.norm(H_tile_x_tile, 2, axis = 1), axis=1)) # row-wise computation of the L2

                # previous, probably more efficient, solution, which gave the wrong results at certain positions
                # A_THHTnnT = utils.sp2cp(A_tile.transpose().tocsr())
                # A_THHTnnT.data[:] = 0
                # kernels_agnn.ahhtnorm_shfl[min(65535, A_tile.shape[0]), 128](A_THHTnnT.data, A_THHTnnT.indices,
                #     A_THHTnnT.indptr, H_tile_y_tile, H_tile_x_tile, H_tile_y_tile_norm, H_tile_x_tile_norm)
                # dM_tile[k:min(k + tau, Z_tile_y_list[l-1].shape[0]), :] += cp.asnumpy(A_THHTnnT @ dZ_tile_tile)
                A_THHTnnT = utils.sp2cp(A_tile)
                A_THHTnnT.data[:] = 0
                kernels_agnn.ahhtnorm_shfl[min(65535, A_tile.shape[0]), 128](A_THHTnnT.data, A_THHTnnT.indices,
                    A_THHTnnT.indptr, H_tile_y_tile, H_tile_x_tile, H_tile_y_tile_norm, H_tile_x_tile_norm)

                # print( MPI.COMM_WORLD.Get_rank(), A_THHTnnT)

                QT = A_THHTnnT.transpose().tocsr()
                dM_tile_tile = QT @ dZ_tile_tile
                dM_tile[k:min(k + tau, Z_tile_y_list[l-1].shape[0]), :] += cp.asnumpy(dM_tile_tile)

                if l > 0:
                    dP_div_dD = utils.sp2cp(A_tile)
                    dP_div_dD.data[:] = 0
                    C_dP_div_D_sq = utils.sp2cp(A_tile)
                    C_dP_div_D_sq.data[:] = 0

                    kernels_agnn.had_div[min(65535, A_tile.shape[0]), 128](C_dP_div_D_sq.data, dP_div_dD.data,
                        dP_div_dD.indices, dP_div_dD.indptr, dZ_tile_tile, H_tile_y_tile, H_tile_x_tile, W_T_gpu, H_tile_y_tile_norm, H_tile_x_tile_norm)
                    dP_div_dD.data = np.nan_to_num(dP_div_dD.data)

                    dCH_tile_tile += dP_div_dD @ H_tile_x_tile
                    dDn_tile_tile += C_dP_div_D_sq @ H_tile_x_tile_norm.reshape(dDn_tile_tile.shape)

                    dP_T_div_dD_T = dP_div_dD.transpose().tocsr()
                    C_dP_div_D_sq_T = C_dP_div_D_sq.transpose().tocsr()

                    dC_TH_tile[k:min(k + tau, Z_tile_y_list[l-1].shape[0]), :] += cp.asnumpy(dP_T_div_dD_T @ H_tile_y_tile)
                    dD_Tn_tile[k:min(k + tau, Z_tile_y_list[l-1].shape[0]), :] += cp.asnumpy(C_dP_div_D_sq_T @ H_tile_y_tile_norm)

            if l > 0:
                dCH_tile[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(dCH_tile_tile)
                dDn_tile[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(dDn_tile_tile)

        bcast_comm.Allreduce(MPI.IN_PLACE, dM_tile, op=MPI.SUM)

        if l > 0:
            # NOTE: probably possible to reduce the number of operations
            if bcast_rank == bcast_root:
                bcast_comm.Reduce(MPI.IN_PLACE, dC_TH_tile, op=MPI.SUM, root=bcast_root)
                bcast_comm.Reduce(MPI.IN_PLACE, dD_Tn_tile, op=MPI.SUM, root=bcast_root)
            else:
                bcast_comm.Reduce(dC_TH_tile, None, op=MPI.SUM, root=bcast_root)
                bcast_comm.Reduce(dD_Tn_tile, None, op=MPI.SUM, root=bcast_root)

            if reduce_rank == reduce_root:
                reduce_comm.Reduce(MPI.IN_PLACE, dCH_tile, op=MPI.SUM, root=reduce_root)
                reduce_comm.Reduce(MPI.IN_PLACE, dDn_tile, op=MPI.SUM, root=reduce_root)
            else:
                reduce_comm.Reduce(dCH_tile, None, op=MPI.SUM, root=reduce_root)
                reduce_comm.Reduce(dDn_tile, None, op=MPI.SUM, root=reduce_root)

            if bcast_rank == bcast_root and reduce_rank == reduce_root:
                dH_1 = dCH_tile + dC_TH_tile
                dH_2 = np.multiply(np.maximum(Z_tile_y_list[l-1], 0), np.divide((dDn_tile + dD_Tn_tile), np.expand_dims(np.linalg.norm(np.maximum(Z_tile_y_list[l-1], 0), 2, axis=1), axis=1)))
                dH_3 = dM_tile @ W_new[l].T
                dH_tile = dH_1 + dH_2 + dH_3
            reduce_comm.Bcast(dH_tile, root=reduce_root)

            dW = np.maximum(Z_tile_x_list[l-1], 0).T @ dM_tile
        else:
            dW = H0_tile_x.T @ dM_tile
        reduce_comm.Allreduce(MPI.IN_PLACE, dW, op=MPI.SUM)

        W_new[l] -= alpha * dW

    return W_new


grid = {
    #     [Px, Py]
    1: [1, 1],
    2: [1, 2],
    4: [2, 2],
    8: [2, 4],
    16: [4, 4],
    32: [4, 8],
    64: [8, 8],
    128: [8, 16],
    256: [16, 16],
    512: [16, 32],
    1024: [32, 32]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AGNN inference')
    parser.add_argument('-d',
                        '--dataset',
                        nargs="?",
                        choices=['random', 'file', 'kronecker'],
                        default='random',
                        help='The source of the adjacency matrix.')
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        nargs="?",
                        default=42,
                        help='The seed for the random number generator.')
    parser.add_argument('-v',
                        '--vertices',
                        type=int,
                        nargs="?",
                        default=100000,
                        help='The number of vertices in the graph.')
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=1000000, help='The number of edges in the graph.')
    parser.add_argument('-t',
                        '--type',
                        nargs="?",
                        choices=['float32', 'float64'],
                        default='float32',
                        help='The type of the data.')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        nargs="?",
                        default=None,
                        help='The file containing the adjacency matrix.')
    parser.add_argument('-l', '--layers', type=int, nargs="?", default=1, help='The number of AGNN layers.')
    parser.add_argument('--features', type=int, nargs="?", default=128, help='The number of features.')
    parser.add_argument('--repeat', type=int, nargs="?", default=5, help='The number of times to repeat the benchmark.')
    parser.add_argument('--warmup', type=int, nargs="?", default=1, help='The number of warmup runs.')
    args = vars(parser.parse_args())

    rng = np.random.default_rng(args['seed'])
    dtype = np.dtype(args['type'])
    num_repeats = args['repeat']
    num_warmup = args['warmup']

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()

    if world_size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")

    # Cartesian grid
    Px, Py = grid[world_size]
    cart_comm = world_comm.Create_cart((Px, Py))
    cart_rank = cart_comm.Get_rank()
    x, y = cart_comm.Get_coords(cart_rank)

    # Subcommunicators
    bcast_comm = cart_comm.Sub([True, False])
    bcast_rank = bcast_comm.Get_rank()
    reduce_comm = cart_comm.Sub([False, True])
    reduce_rank = reduce_comm.Get_rank()

    if args['dataset'] == 'file':
        if args['file'] is None:
            utils.mpi_print(cart_rank, "Please specify the file contaning the adjacency matrix.")
            exit(1)
        absolute_path = os.path.abspath(args['file'])
        if not os.path.exists(absolute_path):
            utils.mpi_print(cart_rank, f"The file {args['file']} does not exist.")
            exit(1)
        folder, filename = os.path.split(absolute_path)
        if not filename.endswith('.npy'):
            utils.mpi_print(cart_rank, f"The file {args['file']} is not a .npy file.")
            exit(1)
        utils.mpi_print(cart_rank, f"Loading adjacency matrix from {args['file']}...")
        kill_signal = np.zeros(1, dtype=np.int32)
        if cart_rank == 0:
            A = utils.load_adjacency_matrix_csr(folder, filename[:-4], row_idx=args['row_index'], dtype=dtype)
            if A.shape[0] != A.shape[1]:
                utils.mpi_print(cart_rank, "The adjacency matrix is not square.")
                kill_signal[0] = 1
        cart_comm.Bcast(kill_signal, root=0)
        if kill_signal[0] == 1:
            exit(1)
    elif args['dataset'] == 'random':
        utils.mpi_print(
            cart_rank,
            f"Generating random adjacency matrix with {args['vertices']} vertices and {args['edges']} edges...")
        if cart_rank == 0:
            A = utils.generate_sparse_matrix(args['vertices'], args['vertices'], args['edges'], dtype, rng)
            A.data[:] = 1.0
    else:
        # args['dataset'] == 'kronecker'
        # Already create the Kronecker graph distributed
        utils.mpi_print(
            cart_rank,
            f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges...")
        args['vertices'], args['edges'], lA = utils.create_kronecker_graph_distributed(args['vertices'], args['edges'], Py, Px, dtype, cart_comm, reduce_comm, rng, True)
        utils.mpi_print(
            cart_rank,
            f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    # Global sizes
    if args['dataset'] == 'kronecker':
        NI = NK = args['vertices']
    else:
        utils.mpi_print(cart_rank, "Broadcasting global sizes...")
        if cart_rank == 0:
            global_sizes = np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64)
        else:
            global_sizes = np.empty(3, dtype=np.int64)
        cart_comm.Bcast(global_sizes, root=0)
        NI, NK, NNZ = global_sizes

    NJ = NL = args['features']

    # Local sizes
    lNI, lNK = int(np.ceil(NI / Px)), int(np.ceil(NK / Py))
    lNJ, lNL = NJ, NL

    if args['dataset'] != 'kronecker':
        # Distribute the adjacency matrix
        utils.mpi_print(cart_rank, "Distributing the adjacency matrix...")
        lA = None
        if cart_rank == 0:
            for i in range(Px):
                for j in range(Py):
                    block = sparse.csr_matrix(A[i * lNI:min(NI, (i + 1) * lNI), j * lNK:min(NK, (j + 1) * lNK)])
                    block.sum_duplicates()
                    block.sort_indices()
                    if x == i and y == j:
                        lA = block
                        lNI = block.shape[0]
                        lNK = block.shape[1]
                        lNNZ = block.nnz
                    else:
                        dst = cart_comm.Get_cart_rank((i, j))
                        size_buffer = np.array([block.shape[0], block.shape[1], block.nnz], dtype=np.int32)
                        cart_comm.Send(size_buffer, dest=dst, tag=0)
                        cart_comm.Send(block.indptr, dest=dst, tag=1)
                        cart_comm.Send(block.indices, dest=dst, tag=2)
                        cart_comm.Send(block.data, dest=dst, tag=3)
            del A
        else:
            size_buffer = np.empty(3, dtype=np.int32)
            cart_comm.Recv(size_buffer, source=0, tag=0)
            lNI, lNK, lNNZ = size_buffer
            indptr = np.empty(lNI + 1, dtype=np.int32)
            indices = np.empty(lNNZ, dtype=np.int32)
            data = np.empty(lNNZ, dtype=dtype)
            cart_comm.Recv(indptr, source=0, tag=1)
            cart_comm.Recv(indices, source=0, tag=2)
            cart_comm.Recv(data, source=0, tag=3)
            lA = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=dtype)

    cart_comm.Barrier()

    # One of the H tiles is replicated in the "bcast" communicators.
    # Therefore, we generate a random block in bcast-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating feature matrix H with shape ({NI}, {NJ})...")
    if cart_rank == 0:
        H_tile_2 = utils.generate_dense_matrix(lNK, lNJ, dtype, rng) * 2 - 1
        for i in range(1, reduce_comm.Get_size(), 1):
            tmp = utils.generate_dense_matrix(lNK, lNJ, dtype, rng) * 2 - 1
            cart_comm.Send(tmp, dest=i, tag=0)
    else:
        if bcast_rank == 0:
            tmp = np.empty((lNK, lNJ), dtype=dtype)
            cart_comm.Recv(tmp, source=0, tag=0)
            H_tile_2 = tmp.reshape((lNK, lNJ))

    if bcast_rank != 0:
        H_tile_2 = np.empty((lNK, lNJ), dtype=dtype)
    utils.bcast_matrix(H_tile_2, bcast_comm, 0)

    # The other H tile is replicated in the "reduce" communicators.
    # Therefore, once the respective tile is received by the process during the previous step,
    # it is broadcast along the "reduce" communicator.
    if reduce_rank == x:
        H_tile_1 = H_tile_2.copy()
    else:
        H_tile_1 = np.empty((lNK, lNJ), dtype=dtype)
    utils.bcast_matrix(H_tile_1, reduce_comm, x)

    # The W matrices are replicated in all ranks.
    # Therefore, we generate random blocks in cart-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating weight matrices W with shape ({NK}, {NL})...")
    W = []
    W_cols = []
    for i in range(args['layers']):
        if cart_rank == 0:
            tmp = utils.generate_dense_matrix(NJ, NL, dtype, rng) * 2 - 1
        else:
            tmp = np.empty((NJ, NL), dtype=dtype)
        cart_comm.Bcast(tmp, root=0)
        W.append(tmp)
        W_cols.append(tmp.shape[1])

    utils.mpi_print(cart_rank, "Generating adjacency matrix blocks...")
    tau, A_blocks = generate_blocks_inference(lA, H_tile_1.shape[1], W_cols)
    utils.mpi_print(cart_rank, f"Tile size: {tau} (rows)")

    output = H_tile_1 #+ np.random.random(size=H_tile_1.shape) * 2 - 1

    # utils.mpi_print(cart_rank, "Computing reference (CPU) output...")
    # ref = cpu_computation(lA, H_tile_1, H_tile_2, W, bcast_comm, reduce_comm, args['layers'], bcast_root=y, reduce_root=x, expected_output_tile=output, alpha=3e-4, grad_of_loss=None)
    # utils.mpi_print(cart_rank, "Computing GPU output...")
    # val_gpu = gpu_computation(lA, A_blocks, H_tile_1, H_tile_2, W, tau, bcast_comm, reduce_comm, args['layers'], bcast_root=y, reduce_root=x, expected_output_tile=output, alpha=3e-4, grad_of_loss=None)
    # utils.mpi_print(cart_rank, "Validating results...")
    # exit()
    # assert np.allclose(ref, val_gpu)

    cart_comm.Barrier()

    # Benchmark
    # utils.mpi_print(cart_rank, "Benchmarking on CPU...")
    # cpu_runtimes = repeat("cpu_computation(lA, H_tile_1, H_tile_2, W, bcast_comm, reduce_comm, args['layers'], bcast_root=y, reduce_root=x, expected_output_tile=output, alpha=3e-4, grad_of_loss=None); cart_comm.Barrier()",
    #                       repeat=1,
    #                       number=1,
    #                       globals={
    #                           **locals(),
    #                           **globals()
    #                       })
    # utils.mpi_print(cart_rank,
    #                 f"CPU: {utils.time_to_ms(np.median(cpu_runtimes))} +- {utils.time_to_ms(np.std(cpu_runtimes))}")
    utils.mpi_print(cart_rank, "Benchmarking on GPU...")
    gpu_stmt = "gpu_computation(lA, A_blocks, H_tile_1, H_tile_2, W, tau, bcast_comm, reduce_comm, args['layers'], bcast_root=y, reduce_root=x, expected_output_tile=output, alpha=3e-4, grad_of_loss=None); cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
    gpu_setup = "cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
    gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=num_warmup+num_repeats, number=1, globals={**locals(), **globals()})
    utils.mpi_print(cart_rank,
                     f"GPU: {utils.time_to_ms(np.median(gpu_runtimes[num_warmup:]))} +- {utils.time_to_ms(np.std(gpu_runtimes[num_warmup:]))}")

    
    # Logging the results
    modelname = 'AGNN'
    filename = 'AGNN_distr_results.csv'
    task = 'training'
    num_layers = args['layers']
    if cart_rank == 0:
        with open(filename, 'a') as f:
            # modelname, inference/training, num_nodes, dtype, Vertices, Edges, num_layers, feature_dim, time, std.
            f.write(f'{modelname}\t{task}\t{world_size}\t{dtype}\t{NI}\t{NNZ}\t{num_layers}\t{NJ}\t{utils.time_to_ms(np.median(gpu_runtimes[num_warmup:]))}\t{utils.time_to_ms(np.std(gpu_runtimes[num_warmup:]))}\n')

