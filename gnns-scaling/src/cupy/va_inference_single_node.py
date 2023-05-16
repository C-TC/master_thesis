import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import kernels
import utils

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
    # Layer computation is σ((A * (H @ H^T)) @ (H @ W))
    # A: NI x NK (sparse)
    # H: NK x NJ (dense)
    # W: NJ x NL (dense)
    # For simplicity, we assume that max(NJ) == max(NL) == max_cols, and that max_cols << NI, NK
    # A: NI x NK (sparse)
    # H: NK x max_cols (dense)
    # W: max_cols x max_cols (dense)
    # We perform the computation in blocks by tiling each of {NI, NK} in T tiles. We need the following blocks:
    # A: T x T (sparse)
    # H_1: T x max_cols (dense)
    # H_2: T x max_cols (dense)
    # A * (H_1 @ H_2^T): T x T (sparse)
    # W: max_cols x max_cols (dense)
    # H_2 @ W: T x max_cols (dense)
    # (A * (H_1 @ H_2^T)) @ (H_2 @ W): T x max_cols (dense)
    # out: T x max_cols (dense)
    # We require the following memory:
    # A: density*dtype*T^2 (data) + density*4*T^2 (indices) + 4*T (indptr)
    # H_1: dtype*T*max_cols
    # H_2: dtype*T*max_cols
    # W: dtype*max_cols*max_cols
    # A * (H_1 @ H_2^T): density*dtype*T^2 (data) + density*4*T^2 (indices) + 4*T (indptr)
    # H_2 @ W: dtype*T*max_cols
    # (A * (H_1 @ H_2^T)) @ (H_2 @ W): dtype*T*max_cols
    # out: dtype*T*max_cols
    # Total: T^2*(2*density*dtype + density*8) + T*(8 + 5*dtype*max_cols) + max_cols^2*dtype

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    max_cols = max(H0_cols, max(W_cols))
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    alpha = 2 * density * dtype + density * 8
    beta = 8 + 5 * dtype * max_cols
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


def cpu_computation(A: sparse.csr_matrix, H_in: np.ndarray, W: List[np.ndarray], layers: int) -> np.ndarray:
    """ Performs computations of all layers on the CPU.

    :param A: The adjacency matrix.
    :param H_in: The initial H matrix.
    :param W: The W matrices.
    :param layers: The number of layers for vanilla attention inference.
    :return: The output (new H matrix after the last layer).
    """

    H = H_in.copy()
    for i in range(layers):
        H = np.maximum(A.multiply(H @ H.T) @ (H @ W[i]), 0)

    return H


def gpu_computation(A: sparse.csr_matrix, A_blocks: List[sparse.csr_matrix], H_in: np.ndarray, W: List[np.ndarray],
                    tau: int, layers: int) -> np.ndarray:
    """ Performs computations of all layers on the CPU.

    :param A: The adjacency matrix.
    :param A_blocks: The blocks of A.
    :param H_in: The initial H matrix.
    :param W: The W matrices.
    :param tau: The tile size.
    :param layers: The number of layers for vanilla attention inference.
    :return: The output (new H matrix after the last layer).
    """

    H = H_in.copy()
    for l in range(layers):
        out = np.zeros((A.shape[0], W[l].shape[1]), dtype=W[l].dtype)

        W_gpu = cp.asarray(W[l])
        block_idx = 0
        for i in range(0, A.shape[0], tau):
            out_tile = cp.asarray(out[i:min(i + tau, A.shape[0]), :])
            H_tile_1 = cp.asarray(H[i:min(i + tau, H.shape[0]), :])
            for k in range(0, H.shape[0], tau):
                tmp = A_blocks[block_idx]
                H_tile_2 = cp.asarray(H[k:min(k + tau, H.shape[0]), :])
                AHHT = utils.sp2cp(tmp)
                AHHT.data[:] = 0
                # kernels.ahht[min(65535, tmp.shape[0]), 128](AHHT.data, AHHT.indices, AHHT.indptr, H_tile_1, H_tile_2)
                kernels.ahht_shfl[min(65535, tmp.shape[0]), 128](AHHT.data, AHHT.indices, AHHT.indptr, H_tile_1, H_tile_2)
                out_tile += AHHT @ (H_tile_2 @ W_gpu)
                block_idx += 1
            out[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(cp.maximum(out_tile, 0))
        if l < layers - 1:
            H = out.copy()

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='vanilla attention inference')
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
    parser.add_argument('-l', '--layers', type=int, nargs="?", default=1, help='The number of vanilla attention layers.')
    parser.add_argument('--features', type=int, nargs="?", default=128, help='The number of features.')
    args = vars(parser.parse_args())

    rng = np.random.default_rng(args['seed'])
    dtype = np.dtype(args['type'])

    if args['dataset'] == 'file':
        if args['file'] is None:
            print("Please specify the file contaning the adjacency matrix.")
            exit(1)
        absolute_path = os.path.abspath(args['file'])
        if not os.path.exists(absolute_path):
            print(f"The file {args['file']} does not exist.")
            exit(1)
        folder, filename = os.path.split(absolute_path)
        if not filename.endswith('.npy'):
            print(f"The file {args['file']} is not a .npy file.")
            exit(1)
        print(f"Loading adjacency matrix from {args['file']}...")
        A = utils.load_adjacency_matrix_csr(folder, filename[:-4], row_idx=args['row_index'], dtype=dtype)
        if A.shape[0] != A.shape[1]:
            print("The adjacency matrix is not square.")
            exit(1)
    elif args['dataset'] == 'random':
        print(f"Generating random adjacency matrix with {args['vertices']} vertices and {args['edges']} edges...")
        A = utils.generate_sparse_matrix(args['vertices'], args['vertices'], args['edges'], dtype, rng)
        A.data[:] = 1.0
    else:
        # args['dataset'] == 'kronecker'
        print(f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges...")
        args['vertices'], args['edges'], A = utils.create_kronecker_graph(args['vertices'], args['edges'], dtype, rng, True)
        print(f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    NK = A.shape[0]
    NJ = NL = args['features']
    NNZ = A.nnz

    print(f"Generating feature matrix H with shape ({NK}, {NJ})...")
    H = utils.generate_dense_matrix(NK, NJ, dtype, rng)
    print(f"Generating weight matrices W with shape ({NJ}, {NL})...")
    W = []
    W_cols = []
    for i in range(args['layers']):
        W.append(utils.generate_dense_matrix(NJ, NL, dtype, rng))
        W_cols.append(W[i].shape[1])

    print("Generating adjacency matrix blocks...")
    tau, A_blocks = generate_blocks_inference(A, H.shape[1], W_cols)
    print(f"Tile size: {tau} (rows)")

    print("Computing reference (CPU) output...")
    ref = cpu_computation(A, H, W, args['layers'])
    print("Computing GPU output...")
    val_gpu = gpu_computation(A, A_blocks, H, W, tau, args['layers'])
    print("Validating results...")
    assert np.allclose(ref, val_gpu)

    # Benchmark
    print("Benchmarking on CPU...")
    cpu_runtimes = repeat("cpu_computation(A, H, W, args['layers'])", repeat=1, number=1, globals={**locals(), **globals()})
    print(f"CPU: {utils.time_to_ms(np.median(cpu_runtimes))} +- {utils.time_to_ms(np.std(cpu_runtimes))}")
    print("Benchmarking on GPU...")
    gpu_stmt = "gpu_computation(A, A_blocks, H, W, tau, args['layers']); cp.cuda.get_current_stream().synchronize()"
    gpu_setup = "cp.cuda.get_current_stream().synchronize()"
    gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=1, number=1, globals={**locals(), **globals()})
    print(f"GPU: {utils.time_to_ms(np.median(gpu_runtimes))} +- {utils.time_to_ms(np.std(gpu_runtimes))}")
