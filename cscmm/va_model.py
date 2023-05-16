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
from copy import deepcopy

import gnn_model
from gnn_model import Parameter
import time


def generate_blocks_inference(
        A: sparse.csr_matrix,
        feature_dim: int) -> Tuple[int, List[List[sparse.csr_matrix]], List[List[List[np.ndarray]]]]:
    """ Generates the blocks for inference.

    The method splits the computation to batches by tiling the adjaceney matrix in blocks of size T x T. The tile size T is
    computed based on the available GPU memory

    :param A: The adjacency matrix.
    :param feature_dim: The number of columns in the initial H matrix.
    :return: The tile size T and the blocks of A.

    Only one kernel: M_block_data, indices, indptr, H_tile_1, Z_tile_1, H_tile_2, N[0:shape[1],:]
    size:   density * tau * tau, density * tau * tau, tau + 1, tau * feature_dim, tau * feature_dim, tau * feature_dim, tau * feature_dim
    nbytes: tau^2 * (density * dtype + density * 4) + tau * (4 * feature_dim * dtype + 4) + 4
    """

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    alpha = density * dtype + density * 4
    beta = 4 * feature_dim * dtype + 4
    gamma = 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau = int(np.ceil((-beta + abs(delta)) / (2 * alpha)))

    tau = min(tau, max(A.shape))
    return utils.generate_blocks_from_tau(A, tau, False)


def generate_blocks_training(
        A: sparse.csr_matrix,
        feature_dim: int) -> Tuple[int, List[List[sparse.csr_matrix]], List[List[List[np.ndarray]]]]:
    """ Generates the blocks for training.

    The method splits the computation to batches by tiling the adjaceney matrix in blocks of size T x T. The tile size T is
    computed based on the available GPU memory

    :param A: The adjacency matrix.
    :param feature_dim: The number of columns in the initial H matrix.
    :return: The tile size T and the blocks of A.

    Backward:
    dN = M^T @ dZ: dN_tile_dev, M_block_data, indices, indptr

    dC = A * (dZ @ N^T): dZ_tile_1_dev, dC_block_data, indices, indptr, N[0:shape[1],:]

    dH += dC @ H: dH_tile_1_dev, dC_block_data, indices, indptr, H[0:shape[1],:]

    dH += dC.T @ H: dH_tile_dev, dC_block_data, indices, indptr, H[0:shape[0],:]

    All strictly smaller than the forward kernel:
    nbytes: tau^2 * (density * dtype + density * 4) + tau * (4 * feature_dim * dtype + 4) + 4
    """
    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    alpha = density * dtype + density * 4
    beta = 4 * feature_dim * dtype + 4
    gamma = 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau = int(np.ceil((-beta + abs(delta)) / (2 * alpha)))

    tau = min(tau, max(A.shape))
    return utils.generate_blocks_from_tau(A, tau, True)


class VAconv(gnn_model.GnnLayer):
    def __init__(self, in_channel: int, out_channel: int, A_shape: np.ndarray, tau: int, use_gpu: bool) -> None:
        super().__init__(in_channel, out_channel, use_gpu)

        self.A_shape = A_shape
        self.tau = tau

        self.timing_data = []

    def init_parameters(self, rng, dtype, cache_grad: bool = True):
        self.parameters.W = Parameter(
            utils.generate_dense_matrix(self.in_channel, self.out_channel, dtype, rng) - 0.5, cache_grad)

    def forward_cpu(self, A: sparse.csr_matrix, H: np.ndarray) -> np.ndarray:
        """Forward pass of the layer on CPU.
        
            :param A: The adjacency matrix.
            :param H: The input H matrix.
            :return: The output matrix.
        """
        M = A.multiply(H[0:A.shape[0], :] @ H[0:A.shape[0], :].T)
        N = np.zeros((H.shape[0], self.out_channel), dtype=H.dtype)
        N[0:A.shape[0], :] = H[0:A.shape[0], :] @ self.W
        Z = np.zeros_like(N)
        Z[0:A.shape[0], :] = np.ascontiguousarray(M @ N[0:A.shape[0], :])

        if self.cache_data:
            self.ctx.H = H
            self.ctx.M = M
            self.ctx.N = N
            self.ctx.Z = Z

        return np.maximum(Z, 0)

    def forward_gpu(self, A_mapping_blocks: Tuple, H: np.ndarray) -> np.ndarray:
        """Forward pass of the layer on GPU.
        
            :param A_mapping_blocks: Tuple of A blocks and mapping blocks.
            :param H: The input H matrix.
            :return: The output matrix.
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        A_dim = max(self.A_shape)

        # N = H @ W
        N = np.zeros((A_dim, self.out_channel), dtype=H.dtype)
        N[0:self.A_shape[0], :] = cp.asnumpy(cp.asarray(H[0:self.A_shape[0], :]) @ cp.asarray(self.W))

        # M = A * (H @ H^T), Z = M @ N
        M_data_blocks = []
        Z = np.zeros_like(N)
        out = np.zeros_like(N)
        for i in range(0, self.A_shape[0], self.tau):
            M_data_i = []
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            H_tile_1 = cp.asarray(H[i:i + A_block_shape_0, :])
            Z_tile_1 = cp.zeros_like(H_tile_1)
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                H_tile_2 = cp.asarray(H[k:k + A_block.shape[1], :])
                M_block_data = cp.zeros_like(A_block.data)
                indices_dev = cp.asarray(A_block.indices)
                indptr_dev = cp.asarray(A_block.indptr)
                kernels.ahht_shfl[min(65535, A_block.shape[0]), 128](M_block_data, indices_dev, indptr_dev, H_tile_1,
                                                                     H_tile_2)

                M_data_i.append(cp.asnumpy(M_block_data))

                # Z
                M_block_dev = cp.sparse.csr_matrix((M_block_data, indices_dev, indptr_dev), shape=A_block.shape)
                Z_tile_1 += M_block_dev @ cp.asarray(N[k:k + A_block.shape[1], :])

            M_data_blocks.append(M_data_i)
            Z[i:i + A_block_shape_0, :] = cp.asnumpy(Z_tile_1)
            out[i:i + A_block_shape_0, :] = cp.asnumpy(cp.maximum(Z_tile_1, 0))

        if self.cache_data:
            self.ctx.H = H
            self.ctx.M_data_blocks = M_data_blocks
            self.ctx.N = N
            self.ctx.Z = Z

        return out

    def backward_cpu(self, A: sparse.csr_matrix, grad_out: np.ndarray):
        """ Backward pass of VA layer on CPU, only for debugging purposes.
            
            param A: adjacency matrix
            param grad_out: gradient of output
        """
        A_dim = max(self.A_shape)
        # relu
        dZ = grad_out * (self.ctx.Z > 0)

        # dN = M^T @ dZ
        dN = np.zeros_like(self.ctx.N)
        dN[0:A.shape[1], :] = self.ctx.M.T @ dZ[0:A.shape[0], :]

        # dH += dN @ W^T
        dH = np.zeros_like(self.ctx.H)
        dH += dN @ self.W.T

        # dC = A * (dZ @ N^T)
        dC = A.multiply(dZ[0:A.shape[0], :] @ self.ctx.N[0:A.shape[1], :].T)

        # dH += dC @ H + dC.T @ H
        dH[0:A.shape[0], :] += dC @ self.ctx.H[0:A.shape[1], :] + dC.T @ self.ctx.H[0:A.shape[0], :]

        # dW = H^T @ dN
        dW = self.ctx.H[0:A.shape[0], :].T @ dN[0:A.shape[1], :]

        return dH

    def backward_gpu(self, A_mapping_blocks: Tuple, grad_out: np.ndarray):
        """ Backward pass of VA layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param grad_out: gradient of output
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        mapping_blocks = A_mapping_blocks[1]
        A_dim = max(self.A_shape)
        start_time = time.perf_counter()
        cscmm_time = 0.0

        # relu
        dZ = grad_out * (self.ctx.Z > 0)
        self.ctx.Z = None

        # dN = M^T @ dZ
        time_0 = time.perf_counter()
        dN = np.zeros_like(self.ctx.N)
        for k in range(0, self.A_shape[1], self.tau):
            A_block_shape_1 = A_blocks[0][k // self.tau].shape[1]
            dN_tile_dev = cp.asarray(dN[k:k + A_block_shape_1, :])
            for i in range(0, self.A_shape[0], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                mapping_block = mapping_blocks[i // self.tau][k // self.tau]
                M_block_data = self.ctx.M_data_blocks[i // self.tau][k // self.tau]
                M_T_block_dev = cp.sparse.csr_matrix((cp.asarray(M_block_data)[cp.asarray(
                    mapping_block[0])], cp.asarray(mapping_block[1]), cp.asarray(mapping_block[2])),
                                                     shape=(A_block.shape[1], A_block.shape[0]))
                dN_tile_dev += M_T_block_dev @ cp.asarray(dZ[i:i + A_block.shape[0], :])
            dN[k:k + A_block_shape_1, :] = cp.asnumpy(dN_tile_dev)
        cscmm_time += time.perf_counter() - time_0
        M_T_block_dev = None
        dN_tile_dev = None
        self.ctx.M_data_blocks = None

        # dW = H.T @ dN
        self.parameters.W.accumulate_grad(
            cp.asnumpy(cp.asarray(self.ctx.H[0:self.A_shape[0], :]).T @ cp.asarray(dN[0:self.A_shape[0], :])))

        if self.is_first_layer:
            # shortcut
            # free memory
            self.ctx.N = None
            self.ctx.H = None
            return None

        dH = np.zeros_like(self.ctx.H)

        # dH += dN @ W.T
        dH = cp.asnumpy(
            cp.asarray(dH[0:self.A_shape[0], :]) + cp.asarray(dN[0:self.A_shape[0], :]) @ cp.asarray(self.W).T)
        dN = None

        # dC = A * (dZ @ N^T)
        dC_data_blocks = []
        for i in range(0, self.A_shape[0], self.tau):
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            dZ_tile_1_dev = cp.asarray(dZ[i:i + A_block_shape_0, :])
            dC_data_i = []
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                dC_block_data = cp.zeros_like(A_block.data)
                kernels.ahht_shfl[min(65535, A_block.shape[0]), 128](dC_block_data, cp.asarray(A_block.indices),
                                                                     cp.asarray(A_block.indptr), dZ_tile_1_dev,
                                                                     cp.asarray(self.ctx.N[k:k + A_block.shape[1], :]))
                dC_data_i.append(cp.asnumpy(dC_block_data))

            dC_data_blocks.append(dC_data_i)

        self.ctx.N = None
        dZ_tile_1_dev = None
        dC_block_data = None

        # dH += dC @ H
        for i in range(0, self.A_shape[0], self.tau):
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            dH_tile_1_dev = cp.asarray(dH[i:i + A_block_shape_0, :])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                dC_block_data = dC_data_blocks[i // self.tau][k // self.tau]
                dC_block_dev = cp.sparse.csr_matrix(
                    (cp.asarray(dC_block_data), cp.asarray(A_block.indices), cp.asarray(A_block.indptr)),
                    shape=A_block.shape)
                dH_tile_1_dev += dC_block_dev @ cp.asarray(self.ctx.H[k:k + A_block.shape[1], :])
            dH[i:i + A_block_shape_0, :] = cp.asnumpy(dH_tile_1_dev)
        dC_block_dev = None
        dH_tile_1_dev = None

        # dH += dC.T @ H
        time_1 = time.perf_counter()
        for k in range(0, self.A_shape[1], self.tau):
            A_block_shape_1 = A_blocks[i // self.tau][0].shape[1]
            dH_tile_dev = cp.asarray(dH[k:k + A_block_shape_1, :])
            for i in range(0, self.A_shape[0], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                mapping_block = mapping_blocks[i // self.tau][k // self.tau]
                dC_block_data = dC_data_blocks[i // self.tau][k // self.tau]
                dC_T_block_dev = cp.sparse.csr_matrix((cp.asarray(dC_block_data)[cp.asarray(
                    mapping_block[0])], cp.asarray(mapping_block[1]), cp.asarray(mapping_block[2])),
                                                      shape=(A_block.shape[1], A_block.shape[0]))
                dH_tile_dev += dC_T_block_dev @ cp.asarray(self.ctx.H[i:i + A_block.shape[0], :])
            dH[k:k + A_block_shape_1, :] = cp.asnumpy(dH_tile_dev)
        cscmm_time += time.perf_counter() - time_1
        dC_T_block_dev = None
        dH_tile_dev = None
        dC_data_blocks = None
        self.ctx.H = None
        end_time = time.perf_counter()
        self.timing_data.extend([cscmm_time, end_time - start_time])

        return dH


class VAmodel(gnn_model.GnnModel):
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 A_shape: List[int],
                 tau: int,
                 use_gpu: bool,
                 num_layers: int,
                 inference_only=False) -> None:
        """Initializes VA model.
            :param in_channels: list of input channels for each layer
            :param out_channel: output channel of last layer
            :param A_shape: shape of adjacency matrix (before tiling)
            :param tau: number of tiles for adjacency matrix
            :param use_gpu: whether to use gpu
            :param num_layers: number of layers
            :param inference_only: whether to use inference only
        """
        assert len(in_channels) == num_layers, "Number of input channels must match number of layers"
        channels = in_channels.copy()
        channels.append(out_channel)
        layers = [VAconv(channels[i], channels[i + 1], A_shape, tau, use_gpu) for i in range(num_layers)]
        super().__init__(layers, inference_only)

    def redistribute_between_layers_forward(self, out):
        return out

    def redistribute_between_layers_backward(self, grad_out):
        return grad_out

    def redistribute_forward_output(self, out):
        return out

    def redistribute_loss_grad(self, grad_out):
        return grad_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VA single node.')
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
                        default=1000,
                        help='The number of vertices in the graph.')
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=50000, help='The number of edges in the graph.')
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
    parser.add_argument('--features', type=int, nargs="?", default=128, help='The number of features.')
    parser.add_argument('--task',
                        choices=['inference', 'training'],
                        default='inference',
                        help='The task to perform, inference or training.')
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
        print(
            f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges..."
        )
        args['vertices'], args['edges'], A = utils.create_kronecker_graph(args['vertices'], args['edges'], dtype, rng,
                                                                          True)
        print(f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    NK = A.shape[0]
    NJ = NL = args['features']
    NNZ = A.nnz

    print(f"Generating feature matrix H with shape ({NK}, {NJ})...")
    H = utils.generate_dense_matrix(NK, NJ, dtype, rng) - 0.5
    grad_out = utils.generate_dense_matrix(NK, NJ, dtype, rng) - 0.5
    print(f"Generating weight matrix W with shape ({NJ}, {NL})...")

    print("Generating adjacency matrix blocks...")
    tau, A_blocks, mapping_blocks = generate_blocks_training(A, NJ)
    print(f"Tile size: {tau} (rows)")

    print("Computing forward reference (CPU) output...")
    VA_cpu = VAconv(NJ, NL, A.shape, tau, False)
    VA_cpu.cache_data = True
    VA_cpu.init_parameters(rng, dtype, True)
    ref_out = VA_cpu.forward(A, H)

    print("Computing GPU output...")
    VA_gpu = VAconv(NJ, NL, A.shape, tau, True)
    VA_gpu.cache_data = True
    VA_gpu.parameters = deepcopy(VA_cpu.parameters)
    gpu_out = VA_gpu.forward((A_blocks, mapping_blocks), H)
    print("Validating results...")
    assert cp.allclose(ref_out, gpu_out, rtol=1e-3, atol=1e-3)
    print("Forward pass validation passed.")

    print("Computing backward reference (CPU) output...")
    ref_grad = VA_cpu.backward(A, grad_out)

    print("Computing GPU output...")
    gpu_grad = VA_gpu.backward((A_blocks, mapping_blocks), grad_out)
    print("Validating results...")
    assert cp.allclose(ref_grad, gpu_grad, rtol=1e-3, atol=1e-3)
    print("Backward pass validation passed.")
