import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import kernels
import kernels_agnn_new
import utils

from mpi4py import MPI
from scipy import sparse
from timeit import repeat
from typing import List, Tuple, Union
from copy import deepcopy

import gnn_model
from gnn_model import Parameter
import agnn_model


class AGNNconvDistr(agnn_model.AGNNconv):
    def __init__(self, in_channel: int, out_channel: int, A_shape: List[int], ceil_A_shape: int, tau: int,
                 use_gpu: bool, bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm, cart_comm: MPI.Cartcomm) -> None:
        """ Initialize AGNN layer.
            param in_channel: number of input channels
            param out_channel: number of output channels
            param A_shape: shape of adjacency matrix on local process
            param ceil_A_shape: maximum shape of adjacency matrix (assume square) on all processes

            Note: ceil_A_shape is used to determine buffer size and avoid variable size MPI communication.
            E.g. processor grid 3x3, global A shape 14x14, then ceil_A_shape = 5
            5x5 5x5 5x4
            5x5 5x5 5x4
            4x5 4x5 4x4
        """

        if use_gpu == False:
            raise ValueError("AGNNconvDistr is only implemented for GPU.")

        super().__init__(in_channel, out_channel, A_shape, tau, use_gpu)

        self.ceil_A_shape = ceil_A_shape

        # communicators
        self.bcast_comm = bcast_comm
        self.reduce_comm = reduce_comm
        self.cart_comm = cart_comm

    def forward_gpu(self, A_mapping_blocks: Tuple, H: np.ndarray) -> np.ndarray:
        """Forward pass of the layer on GPU.
        
            :param A_mapping_blocks: Tuple of A blocks and mapping blocks.
            :param H: The input H matrix.
            :return: The output matrix.
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        A_dim = max(self.A_shape)
        cart_rank = self.cart_comm.Get_rank()
        bcast_rank = self.bcast_comm.Get_rank()
        x, y = self.cart_comm.Get_coords(cart_rank)
        H_tile_1 = H
        H_tile_2 = utils.diagonal_exchange(H, self.cart_comm)


        # M = H @ W
        M_tile_2 = np.zeros((self.ceil_A_shape, self.out_channel), dtype=H.dtype)
        M_tile_2[0:self.A_shape[1], :] = cp.asnumpy(cp.asarray(H_tile_2[0:self.A_shape[1], :]) @ cp.asarray(self.W))

        # Q = A * (H @ H.T) / (n * n.T), Z = Q @ M
        Q_data_blocks = []
        n_tile_1_dev = cp.zeros(self.ceil_A_shape, dtype=H.dtype)
        n_tile_2_dev = cp.zeros(self.ceil_A_shape, dtype=H.dtype)
        n_tile_1_dev[0:self.A_shape[0]] = cp.linalg.norm(cp.asarray(H_tile_1[0:self.A_shape[0], :]), axis=1, ord=2)
        n_tile_2_dev[0:self.A_shape[1]] = cp.linalg.norm(cp.asarray(H_tile_2[0:self.A_shape[1], :]), axis=1, ord=2)
        Z_tile_1 = np.zeros((self.ceil_A_shape, self.out_channel), dtype=H.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            Q_data_i = []
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            Z_seg_tile_1_dev = cp.asarray(Z_tile_1[i:i + A_block_shape_0, :])
            H_seg_tile_1_dev = cp.asarray(H_tile_1[i:i + A_block_shape_0, :])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Q_block_data_dev = cp.zeros_like(A_block.data)
                indices_dev = cp.asarray(A_block.indices)
                indptr_dev = cp.asarray(A_block.indptr)
                kernels_agnn_new.forward_ahhtnorm_shfl[min(65535, A_block.shape[0]),
                                                  128](Q_block_data_dev, indices_dev, indptr_dev, H_seg_tile_1_dev,
                                                       cp.asarray(H_tile_2[k:k + A_block.shape[1], :]),
                                                       n_tile_1_dev[i:i + A_block.shape[0]],
                                                       n_tile_2_dev[k:k + A_block.shape[1]])

                Q_block = cp.sparse.csr_matrix((Q_block_data_dev, indices_dev, indptr_dev), shape=A_block.shape)
                Z_seg_tile_1_dev += Q_block @ cp.asarray(M_tile_2[k:k + A_block.shape[1], :])
                Q_data_i.append(cp.asnumpy(Q_block_data_dev))

            Z_tile_1[i:i + A_block.shape[0], :] = cp.asnumpy(Z_seg_tile_1_dev)
            Q_data_blocks.append(Q_data_i)

        self.reduce_comm.Allreduce(MPI.IN_PLACE, Z_tile_1, MPI.SUM)

        Q_block_data_dev = None
        Q_block = None
        indices_dev = None
        indptr_dev = None
        Z_seg_tile_1_dev = None
        H_seg_tile_1_dev = None

        n_tile_1 = cp.asnumpy(n_tile_1_dev)
        n_tile_2 = cp.asnumpy(n_tile_2_dev)
        n_tile_1_dev = None
        n_tile_2_dev = None

        out_tile_1 = np.maximum(Z_tile_1, 0)

        if self.cache_data:
            self.ctx.H_tile_1 = H_tile_1
            self.ctx.H_tile_2 = H_tile_2
            self.ctx.n_tile_1 = n_tile_1
            self.ctx.n_tile_2 = n_tile_2
            self.ctx.Q_data_blocks = Q_data_blocks
            self.ctx.M_tile_2 = M_tile_2
            self.ctx.Z_tile_1 = Z_tile_1

        return out_tile_1

    def backward_gpu(self, A_mapping_blocks: Tuple, grad_out: np.ndarray):
        """ Backward pass of VA layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param grad_out: gradient of output
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        mapping_blocks = A_mapping_blocks[1]
        A_dim = max(self.A_shape)
        cart_rank = self.cart_comm.Get_rank()
        bcast_rank = self.bcast_comm.Get_rank()
        x, y = self.cart_comm.Get_coords(cart_rank)

        # relu
        dZ_tile_1 = grad_out * (self.ctx.Z_tile_1 > 0)
        self.ctx.Z_tile_1 = None

        # dM = Q^T @ dZ
        dM_tile_2 = np.zeros_like(self.ctx.M_tile_2)
        for k in range(0, self.A_shape[1], self.tau):
            A_block_shape_1 = A_blocks[0][k // self.tau].shape[1]
            dM_seg_tile_2_dev = cp.asarray(dM_tile_2[k:k + A_block_shape_1, :])
            for i in range(0, self.A_shape[0], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                mapping_block = mapping_blocks[i // self.tau][k // self.tau]
                Q_block_data = self.ctx.Q_data_blocks[i // self.tau][k // self.tau]
                Q_T_block_dev = cp.sparse.csr_matrix((cp.asarray(Q_block_data)[cp.asarray(
                    mapping_block[0])], cp.asarray(mapping_block[1]), cp.asarray(mapping_block[2])),
                                                     shape=(A_block.shape[1], A_block.shape[0]))
                dM_seg_tile_2_dev += Q_T_block_dev @ cp.asarray(dZ_tile_1[i:i + A_block.shape[0], :])
            dM_tile_2[k:k + A_block_shape_1, :] = cp.asnumpy(dM_seg_tile_2_dev)

        self.bcast_comm.Allreduce(MPI.IN_PLACE, dM_tile_2, MPI.SUM)
        dM_seg_tile_2_dev = None
        Q_T_block_dev = None
        self.ctx.Q_data_blocks = None

        # dW = H^T @ dM
        dW = cp.asnumpy(
            cp.asarray(self.ctx.H_tile_2[0:self.A_shape[1], :]).T @ cp.asarray(dM_tile_2[0:self.A_shape[1], :]))
        self.reduce_comm.Allreduce(MPI.IN_PLACE, dW, MPI.SUM)
        self.parameters.W.accumulate_grad(dW)

        if self.is_first_layer:
            # shortcut
            # free memory
            self.ctx.H_tile_1 = None
            self.ctx.H_tile_2 = None
            self.ctx.n_tile_1 = None
            self.ctx.n_tile_2 = None
            self.ctx.M_tile_2 = None
            return None

        dH_tile_1 = np.zeros_like(self.ctx.H_tile_1)
        # dH += dM @ W^T
        dH_tile_2_part_1 = np.zeros_like(self.ctx.H_tile_2)
        dH_tile_2_part_1[0:self.A_shape[1], :] += cp.asnumpy(
            cp.asarray(dM_tile_2[0:self.A_shape[1], :]) @ cp.asarray(self.W.T))

        # dQ = A * (dZ @ M.T)
        # dD = - C * dQ  / (D * D), dC = dQ / D
        dC_data_blocks = []
        dD_data_blocks = []
        for i in range(0, self.A_shape[0], self.tau):
            dC_data_i = []
            dD_data_i = []
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            dZ_seg_tile_1_dev = cp.asarray(dZ_tile_1[i:i + A_block_shape_0, :])
            H_seg_tile_1_dev = cp.asarray(self.ctx.H_tile_1[i:i + A_block_shape_0, :])
            n_seg_tile_1_dev = cp.asarray(self.ctx.n_tile_1[i:i + A_block_shape_0])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                dC_block_data = cp.zeros_like(A_block.data)
                dD_block_data = cp.zeros_like(A_block.data)
                kernels_agnn_new.backward_Z_Q_CD_shfl[min(65535, A_block.shape[0]),
                                                 128](dC_block_data, dD_block_data, cp.asarray(A_block.indices),
                                                      cp.asarray(A_block.indptr), dZ_seg_tile_1_dev,
                                                      cp.asarray(self.ctx.M_tile_2[k:k + A_block.shape[1], :]),
                                                      H_seg_tile_1_dev,
                                                      cp.asarray(self.ctx.H_tile_2[k:k + A_block.shape[1], :]),
                                                      n_seg_tile_1_dev,
                                                      cp.asarray(self.ctx.n_tile_2[k:k + A_block.shape[1]]))
                dC_data_i.append(cp.asnumpy(dC_block_data))
                dD_data_i.append(cp.asnumpy(dD_block_data))

            dC_data_blocks.append(dC_data_i)
            dD_data_blocks.append(dD_data_i)
        dZ_seg_tile_1_dev = None
        H_seg_tile_1_dev = None
        n_seg_tile_1_dev = None
        dC_block_data = None
        dD_block_data = None
        self.ctx.M_tile_2 = None

        # dH += dC @ H
        # dn += dD @ n
        dn_tile_1 = np.zeros_like(self.ctx.n_tile_1)
        for i in range(0, self.A_shape[0], self.tau):
            A_block_shape_0 = A_blocks[i // self.tau][0].shape[0]
            dH_seg_tile_1_dev = cp.asarray(dH_tile_1[i:i + A_block_shape_0, :])
            dn_seg_tile_1_dev = cp.asarray(dn_tile_1[i:i + A_block_shape_0])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                indices_dev = cp.asarray(A_block.indices)
                indptr_dev = cp.asarray(A_block.indptr)

                dC_block_data = dC_data_blocks[i // self.tau][k // self.tau]
                dC_block_dev = cp.sparse.csr_matrix((cp.asarray(dC_block_data), indices_dev, indptr_dev),
                                                    shape=A_block.shape)
                dH_seg_tile_1_dev += dC_block_dev @ cp.asarray(self.ctx.H_tile_2[k:k + A_block.shape[1], :])
                dC_block_dev = None

                dD_block_data = dD_data_blocks[i // self.tau][k // self.tau]
                dD_block_dev = cp.sparse.csr_matrix((cp.asarray(dD_block_data), indices_dev, indptr_dev),
                                                    shape=A_block.shape)
                dn_seg_tile_1_dev += dD_block_dev @ cp.asarray(self.ctx.n_tile_2[k:k + A_block.shape[1]])
                dD_block_dev = None

            dH_tile_1[i:i + A_block_shape_0, :] = cp.asnumpy(dH_seg_tile_1_dev)
            dn_tile_1[i:i + A_block_shape_0] = cp.asnumpy(dn_seg_tile_1_dev)

        self.reduce_comm.Allreduce(MPI.IN_PLACE, dH_tile_1, MPI.SUM)
        self.reduce_comm.Allreduce(MPI.IN_PLACE, dn_tile_1, MPI.SUM)

        dH_seg_tile_1_dev = None
        dn_seg_tile_1_dev = None
        indices_dev = None
        indptr_dev = None

        # dH += dC.T @ H
        # dn += dD.T @ n
        dn_tile_2 = np.zeros_like(self.ctx.n_tile_2)
        dH_tile_2_part_2 = np.zeros_like(self.ctx.H_tile_2)
        for k in range(0, self.A_shape[1], self.tau):
            A_block_shape_1 = A_blocks[0][k // self.tau].shape[1]
            dH_seg_tile_2_dev = cp.asarray(dH_tile_2_part_2[k:k + A_block_shape_1, :])
            dn_seg_tile_2_dev = cp.asarray(dn_tile_2[k:k + A_block_shape_1])
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
                dH_seg_tile_2_dev += dC_T_block_dev @ cp.asarray(self.ctx.H_tile_1[i:i + A_block.shape[0], :])
                dC_T_block_dev = None

                dD_block_data = dD_data_blocks[i // self.tau][k // self.tau]
                dD_T_block_dev = cp.sparse.csr_matrix(
                    (cp.asarray(dD_block_data)[T_data_mapping_dev], T_indices_dev, T_indptr_dev),
                    shape=(A_block.shape[1], A_block.shape[0]))
                dn_seg_tile_2_dev += dD_T_block_dev @ cp.asarray(self.ctx.n_tile_1[i:i + A_block.shape[0]])
                dD_T_block_dev = None

            dH_tile_2_part_2[k:k + A_block_shape_1, :] = cp.asnumpy(dH_seg_tile_2_dev)
            dn_tile_2[k:k + A_block_shape_1] = cp.asnumpy(dn_seg_tile_2_dev)

        self.bcast_comm.Allreduce(MPI.IN_PLACE, dH_tile_2_part_2, MPI.SUM)
        self.bcast_comm.Allreduce(MPI.IN_PLACE, dn_tile_2, MPI.SUM)
        # dH += dM @ W.T + dC.T @ H
        dH_tile_2 = dH_tile_2_part_1 + dH_tile_2_part_2
        dH_tile_2_transpose = utils.diagonal_exchange(dH_tile_2, self.cart_comm)
        dn_tile_2_transpose = utils.diagonal_exchange(dn_tile_2, self.cart_comm)
        
        dH_tile_2_part_2 = None
        dn_tile_2 = None
        self.ctx.H_tile_2 = None
        self.ctx.n_tile_2 = None

        dH_seg_tile_2_dev = None
        dn_seg_tile_2_dev = None
        T_data_mapping_dev = None
        T_indices_dev = None
        T_indptr_dev = None

        dH_tile_2_part_1 = None

        dH_tile_1 += dH_tile_2_transpose
        dn_tile_1 += dn_tile_2_transpose

        # dH += H * dn / n
        dH_tile_1[0:self.A_shape[0], :] += self.ctx.H_tile_1[0:self.A_shape[0], :] * (
            dn_tile_1[0:self.A_shape[0]] / self.ctx.n_tile_1[0:self.A_shape[0]])[:, None]

        self.ctx.H_tile_1 = None
        self.ctx.n_tile_1 = None

        return dH_tile_1


class AGNNmodelDistr(gnn_model.GnnModel):
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 A_shape: List[int],
                 ceil_A_shape: int,
                 tau: int,
                 use_gpu: bool,
                 bcast_comm: MPI.Cartcomm,
                 reduce_comm: MPI.Cartcomm,
                 cart_comm: MPI.Cartcomm,
                 num_layers: int,
                 inference_only=False) -> None:
        assert len(in_channels) == num_layers, "Number of input channels must match number of layers"
        channels = in_channels.copy()
        channels.append(out_channel)
        layers = [
            AGNNconvDistr(channels[i], channels[i + 1], A_shape, ceil_A_shape, tau, use_gpu, bcast_comm, reduce_comm,
                          cart_comm) for i in range(num_layers)
        ]
        super().__init__(layers, inference_only)

    def redistribute_between_layers_forward(self, out):
        return out

    def redistribute_between_layers_backward(self, grad_out):
        return grad_out

    def redistribute_forward_output(self, out):
        return out

    def redistribute_loss_grad(self, grad_out):
        return grad_out


grid = {
    #     [Px, Py] assume square grid for now
    1: [1, 1],
    # 2: [1, 2],
    4: [2, 2],
    # 8: [2, 4],
    16: [4, 4],
    # 32: [4, 8],
    64: [8, 8],
    # 128: [8, 16],
    256: [16, 16],
    # 512: [16, 32],
    1024: [32, 32]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AGNN_distr')
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
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=10000, help='The number of edges in the graph.')
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
        utils.mpi_print(
            cart_rank,
            f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges..."
        )
        args['vertices'], args['edges'], A = utils.create_kronecker_graph(args['vertices'], args['edges'], dtype, rng,
                                                                          True)
        utils.mpi_print(
            cart_rank,
            f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    # Global sizes
    utils.mpi_print(cart_rank, "Broadcasting global sizes...")
    if cart_rank == 0:
        global_sizes = np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64)
    else:
        global_sizes = np.empty(3, dtype=np.int64)
    cart_comm.Bcast(global_sizes, root=0)
    NI, NK, NNZ = global_sizes
    NJ = NL = args['features']

    # Local sizes
    # Warning: Do not change this, INI and INK are same on all processes respectively.
    lNI, lNK = int(np.ceil(NI / Px)), int(np.ceil(NK / Py))
    lNJ, lNL = NJ, NL

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
                    block.shape = (min(NI, (x + 1) * lNI) - x * lNI, min(NK, (y + 1) * lNK) - y * lNK)
                    lNNZ = block.nnz
                else:
                    dst = cart_comm.Get_cart_rank((i, j))
                    size_buffer = np.array([block.shape[0], block.shape[1], block.nnz], dtype=np.int32)
                    cart_comm.Send(size_buffer, dest=dst, tag=0)
                    cart_comm.Send(block.indptr, dest=dst, tag=1)
                    cart_comm.Send(block.indices, dest=dst, tag=2)
                    cart_comm.Send(block.data, dest=dst, tag=3)
    else:
        size_buffer = np.empty(3, dtype=np.int32)
        cart_comm.Recv(size_buffer, source=0, tag=0)
        lNNZ = size_buffer[2]
        indptr = np.empty(size_buffer[0] + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=dtype)
        cart_comm.Recv(indptr, source=0, tag=1)
        cart_comm.Recv(indices, source=0, tag=2)
        cart_comm.Recv(data, source=0, tag=3)
        lA = sparse.csr_matrix((data, indices, indptr),
                               shape=(min(NI, (x + 1) * lNI) - x * lNI, min(NK, (y + 1) * lNK) - y * lNK),
                               dtype=dtype)

    cart_comm.Barrier()

    # One of the H tiles is replicated in the "bcast" communicators.
    # Therefore, we generate a random block in bcast-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating feature matrix H with shape ({NI}, {NJ})...")

    if reduce_rank == x:
        H_tile_1 = utils.generate_dense_matrix(lNK, lNJ, dtype, rng) - 0.5
    else:
        H_tile_1 = np.empty((lNK, lNJ), dtype=dtype)
    # reduce_comm.Bcast(H_tile_1, root=x)
    utils.bcast_matrix(H_tile_1, reduce_comm, x)

    # The W matrix are replicated in all ranks.
    # Therefore, we generate random blocks in cart-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating weight matrices W with shape ({NJ}, {NL})...")

    if cart_rank == 0:
        W_local = utils.generate_dense_matrix(NJ, NL, dtype, rng) - 0.5
    else:
        W_local = np.empty((NJ, NL), dtype=dtype)
    cart_comm.Bcast(W_local, root=0)

    utils.mpi_print(cart_rank, "Generating adjacency matrix blocks...")
    tau, A_blocks, mappings = agnn_model.generate_blocks_training(lA, NJ)
    utils.mpi_print(cart_rank, f"Tile size: {tau} (rows)")

    utils.mpi_print(cart_rank, "Computing forward cpu reference...")
    if cart_rank == 0:
        H_global_ref = np.zeros((bcast_comm.size * lNK, NL), dtype=dtype)
    else:
        H_global_ref = None
    if y == 0:
        bcast_comm.Gather(H_tile_1, H_global_ref, root=0)
    if cart_rank == 0:
        AGNN_cpu = agnn_model.AGNNconv(NJ, NL, A.shape, tau, False)
        AGNN_cpu.cache_data = True
        AGNN_cpu.force_set_parameters(cache_grad=True, W=W_local)
        ref_out = AGNN_cpu.forward(A, H_global_ref.copy())

        dist_out = np.zeros_like(ref_out)
    else:
        ref_out = None
        dist_out = None

    AGNN_dist_gpu = AGNNconvDistr(NJ, NL, lA.shape, lNI, tau, True, bcast_comm, reduce_comm, cart_comm)
    AGNN_dist_gpu.cache_data = True
    AGNN_dist_gpu.force_set_parameters(cache_grad=True, W=W_local)
    utils.mpi_print(cart_rank, "Computing distributed forward gpu...")
    local_out = AGNN_dist_gpu.forward((A_blocks, mappings), H_tile_1)
    if y == 0:
        bcast_comm.Gather(local_out, dist_out, root=0)
    if cart_rank == 0:
        assert np.allclose(ref_out, dist_out, atol=1e-2, rtol=1e-2)
        utils.mpi_print(cart_rank, "Correct distributed forward output!")

    utils.mpi_print(cart_rank, "Computing backward cpu reference...")
    if cart_rank == 0:
        ref_grad = AGNN_cpu.backward(A, H_global_ref.copy())
        dist_grad = np.zeros_like(ref_grad)
    else:
        dist_grad = None
        ref_grad = None

    utils.mpi_print(cart_rank, "Computing distributed backward gpu...")
    local_grad = AGNN_dist_gpu.backward((A_blocks, mappings), H_tile_1.copy())
    if y == 0:
        bcast_comm.Gather(local_grad, dist_grad, root=0)
    if cart_rank == 0:
        assert np.allclose(ref_grad, dist_grad, atol=1e-3, rtol=1e-3)
        utils.mpi_print(cart_rank, "Correct distributed backward output!")
