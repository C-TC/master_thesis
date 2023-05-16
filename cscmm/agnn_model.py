import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import kernels
import kernels_agnn_new
import utils

from scipy import sparse
from timeit import repeat
from typing import List, Tuple, Union
from copy import deepcopy

import gnn_model
from gnn_model import Parameter


def generate_blocks_inference(
        A: sparse.csr_matrix,
        feature_dim: int) -> Tuple[int, List[List[sparse.csr_matrix]], List[List[List[np.ndarray]]]]:
    """ Generates the blocks for inference.

    The method splits the computation to batches by tiling the adjaceney matrix in blocks of size T x T. The tile size T is
    computed based on the available GPU memory

    :param A: The adjacency matrix.
    :param feature_dim: The number of columns in the initial H matrix.
    :return: The tile size T and the blocks of A.

    Only one kernel: Q_block_data_dev, indices_dev, indptr_dev, Z_seg_dev, H_seg_tile_1, H_seg_tile_2, M_seg_tile_2, n_dev * 2
    size:   density * tau * tau, density * tau * tau, tau + 1, tau * feature_dim, tau * feature_dim, tau * feature_dim, tau * feature_dim, max(A.shape) * 2
    nbytes: tau^2 * (density * dtype + density * 4) + tau * (4 * feature_dim * dtype + 4) + 4 + 2 * max(A.shape) * dtype
    """

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    alpha = density * dtype + density * 4
    beta = 4 * feature_dim * dtype + 4
    gamma = 4 + 2 * max(A.shape) * dtype - available_memory
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
    1.dM = Q^T @ dZ: 
    arrays: Q_block_data, mapping_data, mapping_indices, mapping_indptr, dM_seg_dev, dZ_seg_dev

    2.dQ = A * (dZ @ M.T), dD = - C * dQ  / (D * D), dC = dQ / D:
    arrays: dC_block_data, dD_block_data, indices_dev, indptr_dev, dZ_seg_dev, M_seg_dev, H_seg_tile_1_dev, H_seg_tile_2_dev, n_seg_tile_1_dev, n_seg_tile_2_dev
    size: density * tau * tau, density * tau * tau, density * tau * tau, tau + 1, tau * feature_dim * 4, tau * 2
    nbytes: tau^2 * (2 * density * dtype + density * 4) + tau * (4 * feature_dim * dtype + 4 + 2 * dtype) + 4

    3.dH += dC @ H, dn += dD @ n
    strictly smaller than dH += dC.T @ H, dn += dD.T @ n

    4.dH += dC.T @ H, dn += dD.T @ n
    arrays: dC_block_data, dC_T_block_data, T_data_mapping_dev, T_indices_dev, T_indptr_dev, dH_seg_tile_2_dev, H_seg_tile_1_dev, dn_seg_tile_2_dev
    size: density * tau * tau * 3, density * tau * tau, tau + 1, tau * feature_dim * 2, tau
    nbytes: tau^2 * (3 * density * dtype + density * 4) + tau * (2 * feature_dim * dtype + 4 + dtype) + 4

    Need to consider case 2 and 4. Both strictly larger than forward kernel.

    """
    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    # case 2
    alpha = 2 * density * dtype + density * 4
    beta = 4 * feature_dim * dtype + 4 + 2 * dtype
    gamma = 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau1 = int(np.ceil((-beta + abs(delta)) / (2 * alpha)))

    # case 4
    alpha = 3 * density * dtype + density * 4
    beta = 2 * feature_dim * dtype + 4 + dtype
    gamma = 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau2 = int(np.ceil((-beta + abs(delta)) / (2 * alpha)))

    tau = min(tau1, tau2, max(A.shape))
    return utils.generate_blocks_from_tau(A, tau, True)


class AGNNconv(gnn_model.GnnLayer):
    def __init__(self, in_channel: int, out_channel: int, A_shape: np.ndarray, tau: int, use_gpu: bool) -> None:
        super().__init__(in_channel, out_channel, use_gpu)

        self.A_shape = A_shape
        self.tau = tau

    def init_parameters(self, rng, dtype, cache_grad: bool = True):
        self.parameters.W = Parameter(
            utils.generate_dense_matrix(self.in_channel, self.out_channel, dtype, rng) - 0.5, cache_grad)

    def forward_cpu(self, A: sparse.csr_matrix, H: np.ndarray) -> np.ndarray:
        """Forward pass of the layer on CPU.
        
            :param A: The adjacency matrix.
            :param H: The input H matrix.
            :return: The output matrix.
        """
        n = np.linalg.norm(H[0:self.A_shape[0], :], axis=1, ord=2)
        Q_data = np.zeros_like(A.data)

        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                Q_data[j] = H[i, :] @ H[A.indices[j], :].T / (n[i] * n[A.indices[j]])

        M = np.zeros((A.shape[0], self.out_channel), dtype=H.dtype)
        M[0:self.A_shape[0], :] = H[0:self.A_shape[0], :] @ self.W

        Q = sparse.csr_matrix((Q_data, A.indices, A.indptr), shape=A.shape)
        Z = np.zeros((H.shape[0], self.out_channel), dtype=H.dtype)
        Z[0:self.A_shape[0], :] = Q @ M[0:self.A_shape[1], :]

        out = np.maximum(Z, 0)

        if self.cache_data:
            self.ctx.H = H
            self.ctx.n = n
            self.ctx.M = M
            self.ctx.Q_data = Q_data
            self.ctx.Z = Z

        return out

    def forward_gpu(self, A_mapping_blocks: Tuple, H: np.ndarray) -> np.ndarray:
        """Forward pass of the layer on GPU.
        
            :param A_mapping_blocks: Tuple of A blocks and mapping blocks.
            :param H: The input H matrix.
            :return: The output matrix.
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        A_dim = max(self.A_shape)

        # M = H @ W
        M = np.zeros((A_dim, self.out_channel), dtype=H.dtype)
        M[0:self.A_shape[0], :] = cp.asnumpy(cp.asarray(H[0:self.A_shape[0], :]) @ cp.asarray(self.W))

        # Q = A * (H @ H.T) / (n * n.T), Z = Q @ M
        Q_data_blocks = []
        n_dev = cp.zeros(A_dim, dtype=H.dtype)
        n_dev[0:self.A_shape[0]] = cp.linalg.norm(cp.asarray(H[0:self.A_shape[0], :]), axis=1, ord=2)
        Z = np.zeros((A_dim, self.out_channel), dtype=H.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            Q_data_i = []
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            Z_seg_dev = cp.asarray(Z[i:i + A_block_shape_0, :])
            H_seg_tile_1_dev = cp.asarray(H[i:i + A_block_shape_0, :])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Q_block_data_dev = cp.zeros_like(A_block.data)
                indices_dev = cp.asarray(A_block.indices)
                indptr_dev = cp.asarray(A_block.indptr)
                kernels_agnn_new.forward_ahhtnorm_shfl[min(65535, A_block.shape[0]),
                                                  128](Q_block_data_dev, indices_dev, indptr_dev, H_seg_tile_1_dev,
                                                       cp.asarray(H[k:k + A_block.shape[1], :]),
                                                       n_dev[i:i + A_block.shape[0]], n_dev[k:k + A_block.shape[1]])

                Q_block = cp.sparse.csr_matrix((Q_block_data_dev, indices_dev, indptr_dev), shape=A_block.shape)
                Z_seg_dev += Q_block @ cp.asarray(M[k:k + A_block.shape[1], :])
                Q_data_i.append(cp.asnumpy(Q_block_data_dev))

            Z[i:i + A_block.shape[0], :] = cp.asnumpy(Z_seg_dev)
            Q_data_blocks.append(Q_data_i)

        Q_block_data_dev = None
        Q_block = None
        indices_dev = None
        indptr_dev = None
        Z_seg_dev = None
        H_seg_tile_1_dev = None

        n = cp.asnumpy(n_dev)
        n_dev = None

        out = np.maximum(Z, 0)

        if self.cache_data:
            self.ctx.H = H
            self.ctx.n = n
            self.ctx.Q_data_blocks = Q_data_blocks
            self.ctx.M = M
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

        # dM = Q^T @ dZ
        dM = np.zeros_like(self.ctx.M)
        Q = sparse.csr_matrix((self.ctx.Q_data, A.indices, A.indptr), shape=A.shape)
        dM[0:self.A_shape[1], :] = Q.T @ dZ[0:self.A_shape[0], :]

        # dW = H^T @ dM
        dW = self.ctx.H[0:self.A_shape[0], :].T @ dM[0:self.A_shape[0], :]

        # dH += dM @ W^T
        dH = np.zeros_like(self.ctx.H)
        dH[0:self.A_shape[0], :] += dM[0:self.A_shape[0], :] @ self.W.T

        # dQ = A * (dZ @ M.T)
        # dD = - C * dQ  / (D * D), dC = dQ / D
        dD_data = np.zeros_like(self.ctx.Q_data)
        dC_data = np.zeros_like(self.ctx.Q_data)
        for i in range(0, len(A.indptr) - 1):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                dQ_data_k = dZ[i, :] @ self.ctx.M[A.indices[k], :].T
                C = self.ctx.H[i, :] @ self.ctx.H[A.indices[k], :].T
                D = self.ctx.n[i] * self.ctx.n[A.indices[k]]
                dD_data[k] = -dQ_data_k * C / D**2
                dC_data[k] = dQ_data_k / D

        # dn = (dD + dD.T) @ n
        dn = np.zeros_like(self.ctx.n)
        dD = sparse.csr_matrix((dD_data, A.indices, A.indptr), shape=A.shape)
        dn += dD @ self.ctx.n[0:self.A_shape[1]] + dD.T @ self.ctx.n[0:self.A_shape[0]]

        # dH += H * (dn / n)
        dH[0:self.A_shape[0], :] += self.ctx.H[0:self.A_shape[0], :] * (dn[0:self.A_shape[0]] /
                                                                        self.ctx.n[0:self.A_shape[0]])[:, None]

        # dH += (dC + dC.T) @ H
        dC = sparse.csr_matrix((dC_data, A.indices, A.indptr), shape=A.shape)
        dH[0:self.A_shape[0], :] += dC @ self.ctx.H[0:self.A_shape[1], :] + dC.T @ self.ctx.H[0:self.A_shape[0], :]

        return dH

    def backward_gpu(self, A_mapping_blocks: Tuple, grad_out: np.ndarray):
        """ Backward pass of VA layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param grad_out: gradient of output
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        mapping_blocks = A_mapping_blocks[1]
        A_dim = max(self.A_shape)

        # relu
        dZ = grad_out * (self.ctx.Z > 0)
        self.ctx.Z = None

        # dM = Q^T @ dZ
        dM = np.zeros_like(self.ctx.M)
        for k in range(0, self.A_shape[1], self.tau):
            A_block_shape_1 = A_blocks[0][k // self.tau].shape[1]
            dM_seg_dev = cp.asarray(dM[k:k + A_block_shape_1, :])
            for i in range(0, self.A_shape[0], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                mapping_block = mapping_blocks[i // self.tau][k // self.tau]
                Q_block_data = self.ctx.Q_data_blocks[i // self.tau][k // self.tau]
                Q_T_block_dev = cp.sparse.csr_matrix((cp.asarray(Q_block_data)[cp.asarray(
                    mapping_block[0])], cp.asarray(mapping_block[1]), cp.asarray(mapping_block[2])),
                                                     shape=(A_block.shape[1], A_block.shape[0]))
                dM_seg_dev += Q_T_block_dev @ cp.asarray(dZ[i:i + A_block.shape[0], :])
            dM[k:k + A_block_shape_1, :] = cp.asnumpy(dM_seg_dev)
        dM_seg_dev = None
        Q_T_block_dev = None
        self.ctx.Q_data_blocks = None

        # dW = H^T @ dM
        dW = cp.asnumpy(cp.asarray(self.ctx.H[0:self.A_shape[0], :]).T @ cp.asarray(dM[0:self.A_shape[0], :]))
        self.parameters.W.accumulate_grad(dW)

        if self.is_first_layer:
            # shortcut
            # free memory
            self.ctx.H = None
            self.ctx.M = None
            self.ctx.n = None
            return None

        # dH += dM @ W^T
        dH = np.zeros_like(self.ctx.H)
        dH[0:self.A_shape[0], :] += cp.asnumpy(cp.asarray(dM[0:self.A_shape[0], :]) @ cp.asarray(self.W.T))

        # dQ = A * (dZ @ M.T)
        # dD = - C * dQ  / (D * D), dC = dQ / D
        dC_data_blocks = []
        dD_data_blocks = []
        for i in range(0, self.A_shape[0], self.tau):
            dC_data_i = []
            dD_data_i = []
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            dZ_seg_dev = cp.asarray(dZ[i:i + A_block_shape_0, :])
            H_seg_tile_1_dev = cp.asarray(self.ctx.H[i:i + A_block_shape_0, :])
            n_seg_tile_1_dev = cp.asarray(self.ctx.n[i:i + A_block_shape_0])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                dC_block_data = cp.zeros_like(A_block.data)
                dD_block_data = cp.zeros_like(A_block.data)
                kernels_agnn_new.backward_Z_Q_CD_shfl[min(65535, A_block.shape[0]),
                                                 128](dC_block_data, dD_block_data, cp.asarray(A_block.indices),
                                                      cp.asarray(A_block.indptr), dZ_seg_dev,
                                                      cp.asarray(self.ctx.M[k:k + A_block.shape[1], :]),
                                                      H_seg_tile_1_dev,
                                                      cp.asarray(self.ctx.H[k:k + A_block.shape[1], :]),
                                                      n_seg_tile_1_dev, cp.asarray(self.ctx.n[k:k + A_block.shape[1]]))
                dC_data_i.append(cp.asnumpy(dC_block_data))
                dD_data_i.append(cp.asnumpy(dD_block_data))

            dC_data_blocks.append(dC_data_i)
            dD_data_blocks.append(dD_data_i)
        dZ_seg_dev = None
        H_seg_tile_1_dev = None
        n_seg_tile_1_dev = None
        dC_block_data = None
        dD_block_data = None
        self.ctx.M = None

        # dH += dC @ H
        # dn += dD @ n
        dn = np.zeros_like(self.ctx.n)
        for i in range(0, self.A_shape[0], self.tau):
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            dH_seg_tile_1_dev = cp.asarray(dH[i:i + A_block_shape_0, :])
            dn_seg_tile_1_dev = cp.asarray(dn[i:i + A_block_shape_0])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                indices_dev = cp.asarray(A_block.indices)
                indptr_dev = cp.asarray(A_block.indptr)

                dC_block_data = dC_data_blocks[i // self.tau][k // self.tau]
                dC_block_dev = cp.sparse.csr_matrix((cp.asarray(dC_block_data), indices_dev, indptr_dev),
                                                    shape=A_block.shape)
                dH_seg_tile_1_dev += dC_block_dev @ cp.asarray(self.ctx.H[k:k + A_block.shape[1], :])
                dC_block_dev = None

                dD_block_data = dD_data_blocks[i // self.tau][k // self.tau]
                dD_block_dev = cp.sparse.csr_matrix((cp.asarray(dD_block_data), indices_dev, indptr_dev),
                                                    shape=A_block.shape)
                dn_seg_tile_1_dev += dD_block_dev @ cp.asarray(self.ctx.n[k:k + A_block.shape[1]])
                dD_block_dev = None

            dH[i:i + A_block_shape_0, :] = cp.asnumpy(dH_seg_tile_1_dev)
            dn[i:i + A_block_shape_0] = cp.asnumpy(dn_seg_tile_1_dev)

        dH_seg_tile_1_dev = None
        dn_seg_tile_1_dev = None
        indices_dev = None
        indptr_dev = None

        # dH += dC.T @ H
        # dn += dD.T @ n
        for k in range(0, self.A_shape[1], self.tau):
            A_block_shape_1 = A_blocks[0][k // self.tau].shape[1]
            dH_seg_tile_2_dev = cp.asarray(dH[k:k + A_block_shape_1, :])
            dn_seg_tile_2_dev = cp.asarray(dn[k:k + A_block_shape_1])
            for i in range(0, self.A_shape[0], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                mapping_block = mapping_blocks[i // self.tau][k // self.tau]
                T_data_mapping_dev = cp.asarray(mapping_block[0])
                T_indices_dev = cp.asarray(mapping_block[1])
                T_indptr_dev = cp.asarray(mapping_block[2])

                dC_block_data = dC_data_blocks[i // self.tau][k // self.tau]
                dC_T_block_dev = cp.sparse.csr_matrix(
                    (cp.asarray(dC_block_data)[T_data_mapping_dev], T_indices_dev, T_indptr_dev),
                    shape=(A_block.shape[1], A_block.shape[0]))
                dH_seg_tile_2_dev += dC_T_block_dev @ cp.asarray(self.ctx.H[i:i + A_block.shape[0], :])
                dC_T_block_dev = None

                dD_block_data = dD_data_blocks[i // self.tau][k // self.tau]
                dD_T_block_dev = cp.sparse.csr_matrix(
                    (cp.asarray(dD_block_data)[T_data_mapping_dev], T_indices_dev, T_indptr_dev),
                    shape=(A_block.shape[1], A_block.shape[0]))
                dn_seg_tile_2_dev += dD_T_block_dev @ cp.asarray(self.ctx.n[i:i + A_block.shape[0]])
                dD_T_block_dev = None

            dH[k:k + A_block_shape_1, :] = cp.asnumpy(dH_seg_tile_2_dev)
            dn[k:k + A_block_shape_1] = cp.asnumpy(dn_seg_tile_2_dev)

        dH_seg_tile_2_dev = None
        dn_seg_tile_2_dev = None
        T_data_mapping_dev = None
        T_indices_dev = None
        T_indptr_dev = None

        # dH += H * dn / n
        dH[0:self.A_shape[0], :] += self.ctx.H[0:self.A_shape[0], :] * (dn[0:self.A_shape[0]] /
                                                                        self.ctx.n[0:self.A_shape[0]])[:, None]

        self.ctx.H = None
        self.ctx.n = None

        return dH


class AGNNmodel(gnn_model.GnnModel):
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 A_shape: List[int],
                 tau: int,
                 use_gpu: bool,
                 num_layers: int,
                 inference_only=False) -> None:
        """Initializes AGNN model.
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
        layers = [AGNNconv(channels[i], channels[i + 1], A_shape, tau, use_gpu) for i in range(num_layers)]
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

    parser = argparse.ArgumentParser(description='AGNN single node.')
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
    AGNN_cpu = AGNNconv(NJ, NL, A.shape, tau, False)
    AGNN_cpu.cache_data = True
    AGNN_cpu.init_parameters(rng, dtype, True)
    ref_out = AGNN_cpu.forward(A, H)

    print("Computing GPU output...")
    AGNN_gpu = AGNNconv(NJ, NL, A.shape, tau, True)
    AGNN_gpu.cache_data = True
    AGNN_gpu.parameters = deepcopy(AGNN_cpu.parameters)
    gpu_out = AGNN_gpu.forward((A_blocks, mapping_blocks), H)
    print("Validating results...")
    assert cp.allclose(ref_out, gpu_out, rtol=1e-3, atol=1e-3)
    print("Forward pass validation passed.")

    print("Computing backward reference (CPU) output...")
    ref_grad = AGNN_cpu.backward(A, grad_out)

    print("Computing GPU output...")
    gpu_grad = AGNN_gpu.backward((A_blocks, mapping_blocks), grad_out)
    print("Validating results...")
    assert cp.allclose(ref_grad, gpu_grad, rtol=1e-3, atol=1e-3)
    print("Backward pass validation passed.")
