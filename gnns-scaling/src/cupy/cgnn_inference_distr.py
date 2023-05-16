import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import utils

from mpi4py import MPI
from scipy import sparse
from timeit import repeat
from typing import List, Tuple, Union


def count_layer_memory(A: sparse.csr_matrix, H_cols, W_cols) -> int:
    """ Counts the memory required for a single inference layer.

    :param A: The adjacency matrix.
    :param H_cols: The number of columns in the H matrix.
    :param W_cols: The number of columns in the W matrix.
    :return: The memory required for a single inference layer.
    """
    # Memory for matrices
    A_memory = A.indptr.nbytes + A.indices.nbytes + A.data.nbytes
    H_memory = A.shape[1] * H_cols * np.dtype(A.dtype).itemsize
    W_memory = H_cols * W_cols * np.dtype(A.dtype).itemsize
    layer_memory = A_memory + H_memory + W_memory
    if H_cols > W_cols:
        # HW = H @ W
        layer_memory += A.shape[1] * W_cols * np.dtype(A.dtype).itemsize
    else:
        # AH = A @ H
        layer_memory += A.shape[0] * H_cols * np.dtype(A.dtype).itemsize
    # tmp = A @ H @ W
    # H1 = np.maximum(tmp, 0)
    layer_memory += 2 * A.shape[0] * W_cols[0] * np.dtype(A.dtype).itemsize
    return layer_memory


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
    # Layer computation is σ(A @ H @ W)
    # A: NI x NK (sparse)
    # H: NK x NJ (dense)
    # W: NJ x NL (dense)
    # For simplicity, we assume that max(NJ) == max(NL) == max_cols, and that max_cols << NI, NK
    # A: NI x NK (sparse)
    # H: NK x max_cols (dense)
    # W: max_cols x max_cols (dense)
    # We perform the computation in blocks by tiling each of {NI, NK} in T tiles. We need the following blocks:
    # A: T x T (sparse)
    # H: T x max_cols (dense)
    # W: max_cols x max_cols (dense)
    # HW: T x max_cols (dense)
    # A @ HW: T x max_cols (dense)
    # out: T x max_cols (dense)
    # We require the following memory:
    # A: density*dtype*T^2 (data) + density*4*T^2 (indices) + 4*T (indptr)
    # H: dtype*T*max_cols
    # W: dtype*max_cols*max_cols
    # HW: dtype*T*max_cols
    # A @ HW: dtype*T*max_cols
    # out: dtype*T*max_cols
    # Total: T^2*(density*dtype + density*4) + T*(4 + 4*dtype*max_cols) + max_cols^2*dtype

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    max_cols = max(H0_cols, max(W_cols))
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]  # * 0.10  # just for local testing

    alpha = density * dtype + density * 4
    beta = 4 + 4 * dtype * max_cols
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


def batch_computation(A: Union[sparse.csr_matrix, cp.sparse.csr_matrix], H: Union[np.ndarray, cp.ndarray],
                      W: Union[np.ndarray, cp.ndarray], partial_out: Union[np.ndarray, cp.ndarray]):
    """ Performs a batch computation.

    :param A: The adjacency matrix.
    :param H: The H matrix.
    :param W: The W matrix.
    :param partial_out: The partial output.
    """
    if H.shape[1] > W.shape[1]:
        partial_out += A @ (H @ W)
    else:
        partial_out += (A @ H) @ W


def layer_cpu(A: sparse.csr_matrix, H: np.ndarray, W: np.ndarray, reduce_comm: MPI.Cartcomm) -> np.ndarray:
    """ Performs a layer computation on the CPU.

    :param A: The adjacency matrix.
    :param H: The H matrix.
    :param W: The W matrix.
    :param reduce_comm: The reduce communicator.
    :return: The output (new H matrix).
    """
    if H.shape[1] > W.shape[1]:
        tmp = A @ (H @ W)
    else:
        tmp = (A @ H) @ W
    reduce_comm.Allreduce(MPI.IN_PLACE, [tmp, tmp.shape[0] * tmp.shape[1]], op=MPI.SUM)
    return np.maximum(tmp, 0)


def layer_gpu(A: sparse.csr_matrix, A_blocks: List[sparse.csr_matrix], H: np.ndarray, W: np.ndarray, tau: int,
              reduce_comm: MPI.Cartcomm) -> np.ndarray:
    """ Performs a layer computation on the GPU.

    :param A: The adjacency matrix.
    :param A_blocks: The blocks of A.
    :param H: The H matrix.
    :param W: The W matrix.
    :param tau: The tile size.
    :return: The output (new H matrix).
    """

    out = np.zeros((A.shape[0], W.shape[1]), dtype=W.dtype)

    W_gpu = cp.asarray(W)
    block_idx = 0
    for i in range(0, A.shape[0], tau):
        out_tile = cp.asarray(out[i:min(i + tau, A.shape[0]), :])
        for k in range(0, H.shape[0], tau):
            tmp = A_blocks[block_idx]
            A_tile = cp.sparse.csr_matrix((cp.asarray(tmp.data), cp.asarray(tmp.indices), cp.asarray(tmp.indptr)),
                                          shape=tmp.shape,
                                          dtype=tmp.dtype)
            H_tile = cp.asarray(H[k:min(k + tau, H.shape[0]), :])
            batch_computation(A_tile, H_tile, W_gpu, out_tile)
            block_idx += 1
        # NOTE: Assuming here a non-CUDA-aware MPI implementation.
        out[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(out_tile)
    reduce_comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)
    out = np.maximum(out, 0)

    return out


def cpu_computation(lA: sparse.csr_matrix, H: np.ndarray, W: List[np.ndarray], bcast_comm: MPI.Cartcomm,
                    reduce_comm: MPI.Cartcomm, layers: int, root: int) -> np.ndarray:
    """ Performs computations of all layers on the CPU.

    :param A: The adjacency matrix tile.
    :param H: The H matrix tile.
    :param W: The W matrices.
    :param bcast_comm: The bcast communicator.
    :param reduce_comm: The reduce communicator.
    :param layers: The number of layers for CGNN inference.
    :param root: The column of the process grid.
    :return: The output (new H matrix tile after the last layer).
    """

    out = H.copy()
    for i in range(layers):
        out = layer_cpu(lA, out, W[i], reduce_comm)
        if i < layers - 1:
            utils.bcast_matrix(tmp, bcast_comm, root)

    return out


def gpu_computation(lA: sparse.csr_matrix, A_blocks: List[sparse.csr_matrix], H: np.ndarray, W: List[np.ndarray],
                    tau: int, bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm, layers: int, root: int) -> np.ndarray:
    """ Performs computations of all layers on the GPU
    :param A: The adjacency matrix tile.
    :param A_blocks: The blocks of A.
    :param H: The H matrix tile.
    :param W: The W matrices.
    :param tau: The tile size.
    :param bcast_comm: The bcast communicator.
    :param reduce_comm: The reduce communicator.
    :param layers: The number of layers for CGNN inference.
    :param root: The column of the process grid.
    :return: The output (new H matrix tile after the last layer).
    """

    out = H.copy()
    for i in range(layers):
        out = layer_gpu(lA, A_blocks, out, W[i], tau, reduce_comm)
        if i < layers - 1:
            utils.bcast_matrix(tmp, bcast_comm, root)

    return out


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

    parser = argparse.ArgumentParser(description='CGNN inference')
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
    parser.add_argument('-l', '--layers', type=int, nargs="?", default=1, help='The number of CGNN layers.')
    parser.add_argument('--features', type=int, nargs="?", default=128, help='The number of features.')
    args = vars(parser.parse_args())

    rng = np.random.default_rng(args['seed'])
    dtype = np.dtype(args['type'])

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
            global_sizes = np.array([A.shape[0], A.shape[1]], dtype=np.int64)
        else:
            global_sizes = np.empty(2, dtype=np.int64)
        cart_comm.Bcast(global_sizes, root=0)
        NI, NK = global_sizes

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

    # The H matrix is replicated in the "bcast" communicators.
    # Therefore, we generate a random block in bcast-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating feature matrix H with shape ({NK}, {NJ})...")
    if bcast_rank == 0:
        H = utils.generate_dense_matrix(lNK, lNJ, dtype, rng)
    else:
        H = np.empty((lNK, lNJ), dtype=dtype)
    utils.bcast_matrix(H, bcast_comm, 0)

    # The W matrices are replicated in all ranks.
    # Therefore, we generate random blocks in cart-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating weight matrices W with shape ({NJ}, {NL})...")
    W = []
    W_cols = []
    for i in range(args['layers']):
        if cart_rank == 0:
            tmp = utils.generate_dense_matrix(NJ, NL, dtype, rng)
        else:
            tmp = np.empty((NJ, NL), dtype=dtype)
        cart_comm.Bcast(tmp, root=0)
        W.append(tmp)
        W_cols.append(tmp.shape[1])

    utils.mpi_print(cart_rank, "Generating adjacency matrix blocks...")
    tau, A_blocks = generate_blocks_inference(lA, H.shape[1], W_cols)
    utils.mpi_print(cart_rank, f"Tile size: {tau} (rows)")

    utils.mpi_print(cart_rank, "Computing reference (CPU) output...")
    ref = cpu_computation(lA, H, W, bcast_comm, reduce_comm, args['layers'], y)
    utils.mpi_print(cart_rank, "Computing GPU output...")
    val_gpu = gpu_computation(lA, A_blocks, H, W, tau, bcast_comm, reduce_comm, args['layers'], y)
    utils.mpi_print(cart_rank, "Validating results...")
    assert np.allclose(ref, val_gpu)

    cart_comm.Barrier()

    # Benchmark
    utils.mpi_print(cart_rank, "Benchmarking on CPU...")
    cpu_runtimes = repeat("cpu_computation(lA, H, W, bcast_comm, reduce_comm, args['layers'], y); cart_comm.Barrier()",
                          repeat=1,
                          number=1,
                          globals={
                              **locals(),
                              **globals()
                          })
    utils.mpi_print(cart_rank,
                    f"CPU: {utils.time_to_ms(np.median(cpu_runtimes))} +- {utils.time_to_ms(np.std(cpu_runtimes))}")
    utils.mpi_print(cart_rank, "Benchmarking on GPU...")
    gpu_stmt = "gpu_computation(lA, A_blocks, H, W, tau, bcast_comm, reduce_comm, args['layers'], y); cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
    gpu_setup = "cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
    gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=1, number=1, globals={**locals(), **globals()})
    utils.mpi_print(cart_rank,
                    f"GPU: {utils.time_to_ms(np.median(gpu_runtimes))} +- {utils.time_to_ms(np.std(gpu_runtimes))}")
