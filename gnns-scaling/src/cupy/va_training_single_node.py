import argparse
import os
from decimal import Decimal


import cupy as cp
import numpy as np

from scipy import sparse
from timeit import repeat
from typing import List, Tuple, Union


import kernels
import utils


def generate_blocks_inference(A: sparse.csr_matrix, H0_cols: int,
                              # W_cols: List[int]) -> Tuple[int, List[sparse.csr_matrix]]:
                              W_cols: List[int]) -> Tuple[int, List[List[sparse.csr_matrix]]]:
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
    # H_1 @ H_2^T: T x T (dense)
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
    # HH^T: dtype*T^2
    # A * (H_1 @ H_2^T): density*dtype*T^2 (data) + density*4*T^2 (indices) + 4*T (indptr)
    # H_2 @ W: dtype*T*max_cols
    # (A * (H_1 @ H_2^T)) @ (H_2 @ W): dtype*T*max_cols
    # out: dtype*T*max_cols
    # Total: T^2*(2*density*dtype + density*8+1) + T*(8 + 5*dtype*max_cols) + max_cols^2*dtype

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    max_cols = max(H0_cols, max(W_cols))
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]  # TODO: manually decreasing memory to test blocking

    alpha = 2 * density * dtype + density * 8 + 1
    beta = 8 + 5 * dtype * max_cols
    gamma = max_cols ** 2 * dtype - available_memory
    delta = np.sqrt(beta ** 2 - 4 * alpha * gamma)
    tau = int(np.ceil((-beta + abs(delta)) / (2 * alpha)))  # TODO: manually decreasing memory to test blocking

    num_tiles_i = int(np.ceil(A.shape[0] / tau))
    num_tiles_k = int(np.ceil(A.shape[1] / tau))
    print(f"Using {num_tiles_i}x{num_tiles_k} batches for inference.")

    A_blocks = []
    if tau >= A.shape[0] and tau >= A.shape[1]:
        A_blocks.append([A])
    else:
        for i in range(0, A.shape[0], tau):
            A_blocks_i = []

            for k in range(0, A.shape[1], tau):
                tmp = sparse.csr_matrix(A[i:min(i + tau, A.shape[0]), k:min(k + tau, A.shape[1])])
                tmp.sum_duplicates()
                tmp.sort_indices()

                A_blocks_i.append(tmp)
                # A_blocks.append(tmp)
            A_blocks.append(A_blocks_i)

    return tau, A_blocks  # generate A_T_blocks as well


def layer_gpu(A: sparse.csr_matrix, A_blocks: List[List[sparse.csr_matrix]], H: np.ndarray, W: np.ndarray,
              tau: int) -> np.ndarray:
    """ Performs a layer computation on the GPU.

    :param A: The adjacency matrix.
    :param A_blocks: The blocks of A.
    :param H: The H matrix.
    :param W: The W matrix.
    :param tau: The tile size.
    :return: The output (new H matrix).
    """

    Z = np.zeros((A.shape[0], W.shape[1]), dtype=W.dtype)
    H_loc = H.copy()

    W_gpu = cp.asarray(W.copy())

    for ii, i in enumerate(range(0, A.shape[0], tau)):

        out_tile = cp.asarray(Z[i:min(i + tau, A.shape[0]), :])
        H_tile_1 = cp.asarray(H_loc[i:min(i + tau, H_loc.shape[0]), :])

        for kk, k in enumerate(range(0, H_loc.shape[0], tau)):

            tmp = A_blocks[ii][kk]

            H_tile_2 = cp.asarray(H_loc[k:min(k + tau, H_loc.shape[0]), :])
            AHHT = utils.sp2cp(tmp)
            AHHT.data[:] = 0

            # kernels.ahht[min(65535, tmp.shape[0]), 128](AHHT.data, AHHT.indices, AHHT.indptr, H_tile_1, H_tile_2)
            kernels.ahht_shfl[min(65535, tmp.shape[0]), 128](AHHT.data, AHHT.indices, AHHT.indptr, H_tile_1, H_tile_2)
            out_tile += AHHT @ (H_tile_2 @ W_gpu)

        Z[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(out_tile)

    return Z
    # return A.multiply(H @ H.T) @ (H @ W)

def cpu_computation(A: sparse.csr_matrix, H: np.ndarray, W: List[np.ndarray], layers: int, output: np.ndarray,
                    alpha: float, grad_of_loss) -> List[np.ndarray]:
    """ Performs computations of all layers on the CPU.

    :param A: The adjacency matrix.
    :param H: The initial H matrix.
    :param W: The W matrices.
    :param layers: The number of layers for vanilla attention inference.
    :return: The output (new H matrix after the last layer).
    """

    H0 = H.copy()

    H_list = []
    H_list.append(H0)

    Z_list = []

    W_new = []
    for i in range(len(W)):
        W_new.append(W[i].copy())

    # multiply: sparse element-wise
    # * : dense element-wise

    # forward pass
    for i in range(layers):
        # M = A.multiply(H_list[i] @ H_list[i].T)  # fused(A, H[i], H[i].transpose())
        # N = (H_list[i] @ W_new[i]).squeeze()
        # Z = (M @ N)
        # Z = (A.multiply(H_list[i] @ H_list[i].T) @ H_list[i]) @ W_new[i]
        Z = A.multiply(H_list[i] @ H_list[i].T) @ (H_list[i] @ W_new[i])
        Z_list.append(Z)
        H_list.append(np.maximum(Z, 0))

    if grad_of_loss is None:
        # dH = H_list[-1] @ (output.T @ output) # * np.linalg.norm(H_list[-1]-output, ord='fro')  # pass gradient of the loss
        dH = H_list[-1] - output # * np.linalg.norm(H_loc-output, ord='fro')  # gradient of the MSE loss (loss=||output-H_list[-1]||_F^2) where ||.|| is the Frobenius norm. May be inefficient for testing purposes?
    else:
        dH = grad_of_loss(output, H_list[-1])  # pass gradient of the loss

    # backward pass
    for i in range(layers - 1, -1, -1):

        # N = H_list[i] @ W_new[i]
        # M = A.multiply(H_list[i] @ H_list[i].T)  # fused(A, H[i], H[i].transpose())
        # dZ = (M @ N > 0) * dH  # Is (Z>0) possibly sparse?
        dZ = ((A.multiply(H_list[i] @ H_list[i].T) @ (H_list[i] @ W_new[i]))>0) * dH  # Is (Z>0) possibly sparse?

        # dN = M.T @ dZ  # fused(M.transpose(), dZ)
        dN = A.transpose().multiply(H_list[i] @ H_list[i].T) @ dZ  # possible optimisation with previous step (avoid transposition), check syntax

        if i > 0:
            dC = A.multiply(dZ @ W_new[i].T @ H_list[i].T)  # fused(A, dZ, N.transpose())
            dCH = dC @ H_list[i]
            dC_TH = dC.T @ H_list[i]

            dH = (dCH + dC_TH) + dN @ W_new[i].T

        dW = H_list[i].T @ dN
        W_new[i] -= alpha * dW

    return W_new
    # return {"dW1": dW1, "dH1": dH1, "dCH": dCH, "dC_TH": dC_TH, "dH_init": dH_init, "dN1": dN1} # dZ, dN, dW, W, H0, H_loc


# @profile
def gpu_computation(A: sparse.csr_matrix, A_blocks: List[List[sparse.csr_matrix]],
                    H: np.ndarray, W: List[np.ndarray],
                    tau: int, layers: int, output: np.ndarray, alpha: float, grad_of_loss=None) -> List[np.ndarray]:
    """ Performs computations of all layers on the GPU.

    :param A: The adjacency matrix.
    :param A_blocks: The blocks of A.
    :param H: The initial H matrix.
    :param W: The W matrices.
    :param tau: The tile size.
    :param layers: The number of layers for vanilla attention inference.
    :return: The output (new H matrix after the last layer).
    """

    Z_list = []
    H0 = H.copy()
    H_loc = H.copy()
    W_new = []
    for i in range(len(W)):
        W_new.append(W[i].copy())

    # forward pass
    for i in range(layers):
        Z_tmp = layer_gpu(A, A_blocks, H_loc, W_new[i], tau)
        Z_list.append(Z_tmp)
        H_loc = np.maximum(Z_tmp, 0)

    # default value for grad_of_loss: derivative of the MSE loss (Frobenius norm)
    if grad_of_loss is None:
        # dH = H_loc @ (output.T @ output) # * np.linalg.norm(H_loc-output, ord='fro')  # gradient of the MSE loss (loss=||output-H_list[-1]||_F^2) where ||.|| is the Frobenius norm. May be inefficient for testing purposes?
        dH = H_loc - output # * np.linalg.norm(H_loc-output, ord='fro')  # gradient of the MSE loss (loss=||output-H_list[-1]||_F^2) where ||.|| is the Frobenius norm. May be inefficient for testing purposes?
    else:
        dH = grad_of_loss(output, H_loc)  # pass gradient of the loss # TODO: for testing, use simple difference

    # backward pass
    for l in range(layers - 1, -1, -1):

        W_gpu = cp.asarray(W_new[l])
        W_T_gpu = W_gpu.T

        ######## PART 1: compute dZ and dN #########

        dN = np.zeros((A.shape[0], W_new[l].shape[1]), dtype=W_new[l].dtype)
        dZ = np.zeros((A.shape[0], W_new[l].shape[1]), dtype=W_new[l].dtype)
        if l>0:
            dCH = np.zeros(shape=(A.shape[0], Z_list[l - 1].shape[1]), dtype=A.dtype)  # =(dC)@H
            dC_TH = np.zeros(shape=(A.shape[0], Z_list[l - 1].shape[1]), dtype=A.dtype)  # =(dCT)@H


        ## Tiling of A and H to compute output (=Z) and A^THH^T (=:M_T)
        # outer loop
        for ii, i in enumerate(range(0, A.shape[0], tau)):

            Z_tile_1 = cp.asarray(Z_list[l][i:min(i + tau, A.shape[0]), :])  # fill Z_tile with zeros
            if l > 0:
                H_tile_1 = cp.asarray(np.maximum(Z_list[l-1][i:min(i + tau, A.shape[0]), :],0))
            else:
                H_tile_1 = cp.asarray(H0[i:min(i + tau, A.shape[0]), :])

            dH_tile = cp.asarray(dH[i:min(i + tau, A.shape[0]), :])

            dZ_tile = (Z_tile_1 > 0) * dH_tile

            for kk, k in enumerate(range(0, H_loc.shape[0], tau)):

                tmp_1 = A_blocks[ii][kk].transpose(copy=True).tocsr()
                if l>0:
                    Z_tile_2 = cp.asarray(Z_list[l-1][k:min(k + tau, A.shape[0]), :])  # fill Z_tile with zeros
                    H_tile_2 = cp.maximum(Z_tile_2, 0)
                else:
                    H_tile_2 = cp.asarray(H0[k:min(k + tau, A.shape[0]), :])  # fill Z_tile with zeros

                ATHHT = utils.sp2cp(tmp_1)
                ATHHT.data[:] = 0

                # different kernel versions:
                # kernels.ahht[min(65535, tmp_1.shape[0]), 128](ATHHT.data, ATHHT.indices, ATHHT.indptr, H_tile_2, H_tile_1)
                kernels.ahht_shfl[min(65535, tmp_1.shape[0]), 128](ATHHT.data, ATHHT.indices, ATHHT.indptr, H_tile_2, H_tile_1)
                # ATHHT = utils.sp2cp(tmp_1.multiply(cp.asnumpy(H_tile_1) @  cp.asnumpy(H_tile_2).T).tocsr())


                dN_tile = ATHHT @ dZ_tile
                dN[k:min(k + tau, A.shape[0]), :] += cp.asnumpy(dN_tile)

            # only things we won't use in the loop!

            dZ[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(dZ_tile)

            if l > 0:  # we don't need dH in the first layer (here we compute dH=dH^{(l+1)} )

                ######## PART 2: compute dH #########

                # For reference:
                # dC = A.multiply(dZ @ W[l].T @ H_list[l].T)
                # dH = (dC + dC.T) @ H_list[l] + dN @ W[l].T

                # assuming sparse transposition is expensive, we compute dC and dC_T
                dCH_tile = cp.asarray(dCH[i:min(i + tau, A.shape[0]), :])

                dZ_tile_1 = cp.asarray(dZ[i:min(i + tau, A.shape[0]), :])  # for dC
                H_tile_1 = cp.asarray(np.maximum(Z_list[l-1][i:min(i + tau, A.shape[0]), :], 0))  # for dC_T

                # inner loop
                for kk, k in enumerate(range(0, H_loc.shape[0], tau)):

                    A_tile = A_blocks[ii][kk]

                    # we use the following tile twice: once for computing dC, once for multiplication dC @ H
                    H_tile_2 = cp.asarray(np.maximum(Z_list[l-1], 0)[k:min(k + tau, Z_list[l-1].shape[0]), :])  # for dC

                    A_dZWTHT = utils.sp2cp(A_tile)
                    A_dZWTHT.data[:] = 0

                    # different kernel versions:
                    kernels.masked_dense[min(65535, A_tile.shape[0]), 128](A_dZWTHT.data, A_dZWTHT.indices,
                                                                             A_dZWTHT.indptr, dZ_tile_1, W_T_gpu,
                                                                             H_tile_2)  # maybe feed W instead of W^T
                    # kernels.masked_dense_shfl[min(65535, A_tile.shape[0]), 128](A_dZWTHT.data, A_dZWTHT.indices,
                    #                                                          A_dZWTHT.indptr, dZ_tile_1, W_T_gpu,
                    #                                                          H_tile_2)  # maybe feed W instead of W^T

                    # A_dZWTHT = utils.sp2cp(A_tile.multiply(cp.asnumpy(dZ_tile_1) @ cp.asnumpy(W_T_gpu) @  cp.asnumpy(H_tile_2).T).tocsr())

                    dCH_tile += A_dZWTHT @ H_tile_2
                    # slowest variant: 30 seconds total runtime
                    # A_dZWTHT_T = A_dZWTHT.transpose().tocsr(copy=True)
                    # a little bit faster: 24 seconds total runtime
                    temp = A_dZWTHT.get()
                    temp_t = temp.transpose().tocsr(copy=True)
                    A_dZWTHT_T = utils.sp2cp(temp_t)
                    # even slower variant: 42 seconds total runtime
                    # A_T = A_tile.transpose().tocsr(copy=True)
                    # A_dZWTHT_T = utils.sp2cp(A_T)
                    # A_dZWTHT_T.data[:] = 0
                    # kernels.masked_dense[min(65535, A_T.shape[0]), 128](A_dZWTHT_T.data, A_dZWTHT_T.indices,
                    #     A_dZWTHT_T.indptr, H_tile_2, W_gpu, dZ_tile_1)
                    dC_TH_tile = A_dZWTHT_T @ H_tile_1
                    dC_TH[k:min(k + tau, A.shape[0]), :] += cp.asnumpy(dC_TH_tile)

                # store intermediate result
                dCH[i:min(i + tau, A.shape[0]), :] = cp.asnumpy(dCH_tile)

        if l>0:
            dH = (dCH + dC_TH) + dN @ cp.asnumpy(W_T_gpu)
            dW = np.maximum(Z_list[l-1], 0).T @ dN
        else:
            dW = H0.T @ dN

        W_new[l] -= alpha * dW

    return W_new
    # return {"dW1": dW1, "dH1": dH1, "dCH": dCH, "dC_TH": dC_TH, "dH_init": dH_init, "dN1": dN1} # dZ, dN, dW, W, H0, H_loc

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
                        choices=['float16', 'float32', 'float64'],
                        default='float32',
                        help='The type of the data.')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        nargs="?",
                        default=None,
                        help='The file containing the adjacency matrix.')
    parser.add_argument('-l', '--layers', type=int, nargs="?", default=2,
                        help='The number of vanilla attention layers.')
    parser.add_argument('--features', type=int, nargs="?", default=128, help='The number of features.')
    args = vars(parser.parse_args())

    np.random.seed(args['seed'])

    rng = np.random.default_rng(seed=args['seed'])
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
        A.data = np.ones_like(A.data)
    else:
        # args['dataset'] == 'kronecker'
        print(
            f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges...")
        args['vertices'], args['edges'], A = utils.create_kronecker_graph(args['vertices'], args['edges'], dtype, rng, True)
        print(f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    NK = A.shape[0]
    NJ = NL = args['features']
    NNZ = A.nnz

    print(f"Generating feature matrix H with shape ({NK}, {NJ})...")
    H = utils.generate_dense_matrix(NK, NJ, dtype, rng)*2-1
    print(f"Generating weight matrices W with shape ({NJ}, {NL})...")

    W = []
    W_cols = []
    for i in range(args['layers']):
        W.append(utils.generate_dense_matrix(NJ, NL, dtype, rng)*2-1)
        W_cols.append(W[i].shape[1])

    print("Generating adjacency matrix blocks...")
    tau, A_blocks = generate_blocks_inference(A, H.shape[1], W_cols)
    print(f"Tile size: {tau} (rows)")

    output = H + np.random.random(size=H.shape)*2-1

    # print("Computing reference (CPU) output...")
    # ref = cpu_computation(A=A, H=H, W=W, layers=args['layers'], output=output, alpha=3e-4, grad_of_loss=None)
    # print("Computing GPU output...")
    # val_gpu = gpu_computation(A=A, A_blocks=A_blocks, H=H, W=W, tau=tau, layers=args['layers'], output=output, alpha=3e-4, grad_of_loss=None)
    # print("Validating results...")
    # for i in ref.keys():
    #     print(f"Relative MAE for {i} at precision {args['type']} equals", '%.2E' % Decimal(((np.mean(np.abs((ref[i]-val_gpu[i]))))/(1e-12+np.mean(ref[i]))).astype("float64")))
    # print(f"Relative MAE of weights for {args['layers']} layers at precision {args['type']} equals", '%.2E' % Decimal(np.mean(np.abs(np.concatenate(ref)-np.concatenate(val_gpu))/np.abs(np.mean(ref))).astype("float64")))
    # print(f"Absolute MAE of weights for {args['layers']} layers at precision {args['type']} equals", '%.2E' % Decimal(np.mean(np.abs(np.concatenate(ref)-np.concatenate(val_gpu))).astype("float64")))
    # exit()
    # assert np.allclose(ref, val_gpu, rtol=1e-2), "The GPU output does not match the reference output."
    #  The assertion only succeeds at very low tolerances. Is this caused by rounding errors or an erro in the code?

    # Benchmark
    # print("Benchmarking on CPU...")
    # cpu_runtimes = repeat(
    #     "cpu_computation(A=A, H=H, W=W, layers=args['layers'], output=H, alpha=3e-4, grad_of_loss=None)", repeat=1,
    #     number=1, globals={**locals(), **globals()})
    # print(f"CPU: {utils.time_to_ms(np.median(cpu_runtimes))} +- {utils.time_to_ms(np.std(cpu_runtimes))}")
    print("Benchmarking on GPU...")
    gpu_stmt = "gpu_computation(A=A, A_blocks=A_blocks, H=H, W=W, tau=tau, layers=args['layers'], output=H, alpha=3e-4, grad_of_loss=None); cp.cuda.get_current_stream().synchronize()"
    gpu_setup = "cp.cuda.get_current_stream().synchronize()"
    gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=1, number=1, globals={**locals(), **globals()})
    print(f"GPU: {utils.time_to_ms(np.median(gpu_runtimes))} +- {utils.time_to_ms(np.std(gpu_runtimes))}")

# python va_training_single_node_v2.py --dataset random --vertices 100000 --edges 1000000 --type float32 --layers 2 --features 128
