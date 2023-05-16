import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import kernels
import kernels_gat
import gnn_model
import gat_model
import utils

from mpi4py import MPI
from scipy import sparse
from timeit import repeat
from typing import List, Tuple, Union


class GATconvDistr(gat_model.GATconv):
    def __init__(self, in_channel: int, out_channel: int, A_shape: List[int], ceil_A_shape: int, tau: int,
                 use_gpu: bool, bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm, cart_comm: MPI.Cartcomm) -> None:
        """ Initialize GAT layer.
            param in_channel: number of input channels
            param out_channel: number of output channels
            param A_shape: shape of adjacency matrix on local process
            param ceil_A_shape: maximum shape of adjacency matrix (assume square) on all processes
            param feature_dim: dimension of feature vector

            Note: ceil_A_shape is used to determine buffer size and avoid variable size MPI communication.
            E.g. processor grid 3x3, global A shape 14x14, then ceil_A_shape = 5
            5x5 5x5 5x4
            5x5 5x5 5x4
            4x5 4x5 4x4
        """

        if use_gpu == False:
            raise ValueError("GATconvDistr is only implemented for GPU.")

        super().__init__(in_channel, out_channel, A_shape, tau, use_gpu)

        self.ceil_A_shape = ceil_A_shape

        # communicators
        self.bcast_comm = bcast_comm
        self.reduce_comm = reduce_comm
        self.cart_comm = cart_comm

    def forward_gpu(self, A_mapping_blocks: Tuple, H: np.ndarray):
        """ Forward pass of GAT layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param input: input H matrix, tile_1
            return: output H matrix, tile_1
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        A_dim = max(self.A_shape)
        cart_rank = self.cart_comm.Get_rank()
        bcast_rank = self.bcast_comm.Get_rank()
        x, y = self.cart_comm.Get_coords(cart_rank)
        H_tile_1 = H
        H_tile_2 = utils.diagonal_exchange(H_tile_1, self.cart_comm)

        # M = H @ W
        # TODO: optimize these lines (compute on the fly?)
        M_tile_1 = cp.zeros((self.ceil_A_shape, self.out_channel), dtype=H.dtype)
        M_tile_1[0:self.A_shape[0], :] = cp.asarray(H_tile_1[0:self.A_shape[0], :]) @ cp.asarray(self.W)
        if x == y:
            M_tile_2 = M_tile_1
        else:
            M_tile_2 = cp.zeros((self.ceil_A_shape, self.out_channel), dtype=H.dtype)
            M_tile_2[0:self.A_shape[1], :] = cp.asarray(H_tile_2[0:self.A_shape[1], :]) @ cp.asarray(self.W)
        H_tile_2 = None
        l_tile_1 = cp.asnumpy(M_tile_1[0:self.A_shape[0], :] @ cp.asarray(self.a_l))
        r_tile_2 = cp.asnumpy(M_tile_2[0:self.A_shape[1], :] @ cp.asarray(self.a_r))
        M_tile_1 = cp.asnumpy(M_tile_1)
        if x == y:
            M_tile_2 = M_tile_1
        else:
            M_tile_2 = cp.asnumpy(M_tile_2)

        # E_data_blocks similar to A_blocks but only data of E
        E_data_blocks: List[List[np.ndarray]] = []
        # also compute row_max of E
        E_row_max_dev = cp.full(self.A_shape[0], cp.NINF, dtype=H.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            E_data_blocks_i = []
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                E_data_blocks_i.append(
                    self.gat_forward_gpu_kernel_fuse_E_rowmax(i, k, A_block, None, None, None, l_tile_1, r_tile_2,
                                                              E_row_max_dev))
            E_data_blocks.append(E_data_blocks_i)

        E_row_max_host = cp.asnumpy(E_row_max_dev)
        # Allreduce E_row_max
        self.reduce_comm.Allreduce(MPI.IN_PLACE, E_row_max_host, op=MPI.MAX)
        E_row_max_dev = cp.asarray(E_row_max_host)
        E_row_max_host = None

        # softmax row-wise
        # Alpha before division
        Alpha_data_blocks: List[List[np.ndarray]] = []
        # also compute row_sum of this Alpha
        alpha_row_sum_dev = cp.zeros(self.A_shape[0], dtype=E_row_max_dev.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            Alpha_data_blocks_i = []
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                E_data_block = E_data_blocks[i // self.tau][k // self.tau]
                Alpha_data_blocks_i.append(
                    self.gat_forward_gpu_kernel_softmax_1(i, k, A_block, E_data_block, E_row_max_dev,
                                                          alpha_row_sum_dev))
            Alpha_data_blocks.append(Alpha_data_blocks_i)

        alpha_row_sum_host = cp.asnumpy(alpha_row_sum_dev)
        # Allreduce alpha_row_sum
        self.reduce_comm.Allreduce(MPI.IN_PLACE, alpha_row_sum_host, op=MPI.SUM)
        alpha_row_sum_dev = cp.asarray(alpha_row_sum_host) + cp.finfo(alpha_row_sum_host.dtype).eps
        alpha_row_sum_host = None

        # compute the division in Alpha and Z = Alpha @ M
        Z_tile_1 = np.zeros_like(H_tile_1)
        for i in range(0, self.A_shape[0], self.tau):
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = Alpha_data_blocks[i // self.tau][k // self.tau]
                self.gat_forward_gpu_kernel_softmax_2_matmul_M(i, k, A_block, Alpha_data_block, M_tile_2, Z_tile_1,
                                                               alpha_row_sum_dev)
        # Allreduce Z_tile_1
        self.reduce_comm.Allreduce(MPI.IN_PLACE, Z_tile_1, op=MPI.SUM)

        # relu
        # TODO: fuse this in Z = Alpha @ M?
        output = np.maximum(Z_tile_1, 0)

        if self.cache_data:
            self.ctx.H_tile_1 = H_tile_1
            self.ctx.M_tile_1 = M_tile_1
            self.ctx.M_tile_2 = M_tile_2
            self.ctx.l_tile_1 = l_tile_1
            self.ctx.r_tile_2 = r_tile_2
            self.ctx.Alpha_data_blocks = Alpha_data_blocks
            self.ctx.Z_tile_1 = Z_tile_1

        return output

    def backward_gpu(self, A_mapping_blocks: Tuple, grad_out: np.ndarray):
        """ Backward pass of GAT layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param grad_out: gradient of next layer, tile_1
        """
        A_blocks: List[sparse.csr_matrix] = A_mapping_blocks[0]
        mapping_blocks: List[List[List[np.ndarray]]] = A_mapping_blocks[1]
        A_dim = max(self.A_shape)
        cart_rank = self.cart_comm.Get_rank()
        bcast_rank = self.bcast_comm.Get_rank()
        x, y = self.cart_comm.Get_coords(cart_rank)
        # relu
        dZ_tile_1 = grad_out * (self.ctx.Z_tile_1 > 0)
        # free memory
        self.ctx.Z_tile_1 = None

        # dAlpha = A * (dZ @ M^T)
        dAlpha_blocks: List[List[np.ndarray]] = []
        # dAlpha(i,:) @ Alpha(i,:).T
        dAlpha_dot_Alpha_dev = cp.zeros(self.A_shape[0], dtype=grad_out.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            dAlpha_blocks_i = []
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = self.ctx.Alpha_data_blocks[i // self.tau][k // self.tau]
                dAlpha_blocks_i.append(
                    self.gat_backward_gpu_kernel_Z_Alpha_rowdot(i, k, A_block, Alpha_data_block, dZ_tile_1,
                                                                self.ctx.M_tile_2, dAlpha_dot_Alpha_dev))
            dAlpha_blocks.append(dAlpha_blocks_i)
        # Allreduce dAlpha_dot_Alpha_dev
        dAlpha_dot_Alpha_host = cp.asnumpy(dAlpha_dot_Alpha_dev)
        self.reduce_comm.Allreduce(MPI.IN_PLACE, dAlpha_dot_Alpha_host, op=MPI.SUM)
        dAlpha_dot_Alpha_dev = cp.asarray(dAlpha_dot_Alpha_host)
        dAlpha_dot_Alpha_host = None

        # backprop softmax, recompute D, backprop leaky relu, compute dl, dr
        dl_tile_1 = cp.zeros(self.ceil_A_shape, dtype=grad_out.dtype)
        dr_tile_2 = cp.zeros(self.ceil_A_shape, dtype=grad_out.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = self.ctx.Alpha_data_blocks[i // self.tau][k // self.tau]
                dAlpha_block = dAlpha_blocks[i // self.tau][k // self.tau]
                self.gat_backward_gpu_kernel_softmax_leakyrelu_lr(i, k, A_block, Alpha_data_block, dAlpha_block,
                                                                  self.ctx.l_tile_1, self.ctx.r_tile_2, dl_tile_1,
                                                                  dr_tile_2, dAlpha_dot_Alpha_dev)
        # Allreduce dl_tile_1
        dl_tile_1_host = cp.asnumpy(dl_tile_1)
        self.reduce_comm.Allreduce(MPI.IN_PLACE, dl_tile_1_host, op=MPI.SUM)
        dl_tile_1 = cp.asarray(dl_tile_1_host)
        dl_tile_1_host = None
        # Allreduce dr_tile_2
        dr_tile_2_host = cp.asnumpy(dr_tile_2)
        self.bcast_comm.Allreduce(MPI.IN_PLACE, dr_tile_2_host, op=MPI.SUM)
        dr_tile_2 = cp.asarray(dr_tile_2_host)
        dr_tile_2_host = None

        # free memory
        dAlpha_dot_Alpha_dev = None
        dAlpha_blocks = None
        self.ctx.l_tile_1 = None
        self.ctx.r_tile_2 = None

        # dM = dl @ a_l.T + (dr @ a_r.T + Alpha.T @ dZ)
        dM_tile_1 = np.zeros_like(self.ctx.M_tile_1, dtype=grad_out.dtype)
        # Part1: dl @ a_l.T: tile_1
        dM_tile_1[0:self.A_shape[0], :] += cp.asnumpy(
            dl_tile_1[0:self.A_shape[0], None] @ cp.asarray(self.a_l[None, :]))

        # Part2: dr @ a_r.T + Alpha.T @ dZ: tile_2
        dM_p2_tile_2 = np.zeros_like(self.ctx.M_tile_2, dtype=grad_out.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            dZ_dev = cp.asarray(dZ_tile_1[i:i + A_blocks[i // self.tau][0].shape[0], :])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = self.ctx.Alpha_data_blocks[i // self.tau][k // self.tau]
                mapping_block = mapping_blocks[i // self.tau][k // self.tau]
                Alpha_T_dev = cp.sparse.csr_matrix((cp.asarray(Alpha_data_block)[cp.asarray(
                    mapping_block[0])], cp.asarray(mapping_block[1]), cp.asarray(mapping_block[2])),
                                                   shape=(A_block.shape[1], A_block.shape[0]))
                dM_p2_tile_2[k:k + A_block.shape[1], :] += cp.asnumpy(Alpha_T_dev @ dZ_dev)
                
        self.bcast_comm.Allreduce(MPI.IN_PLACE, dM_p2_tile_2, op=MPI.SUM)
        dM_p2_tile_2[0:self.A_shape[1], :] +=  cp.asnumpy(dr_tile_2[0:self.A_shape[1], None]) @ self.a_r[None, :]
        dM_p2_tile_2_transpose = utils.diagonal_exchange(dM_p2_tile_2, self.cart_comm)

        # Rename dM_p2_tile_2_transpose to dM_p2_tile_1
        dM_p2_tile_1 = dM_p2_tile_2_transpose
        dM_p2_tile_2_transpose = None
        dM_tile_1 += dM_p2_tile_1
        dM_p2_tile_1 = None

        # free memory
        dZ_tile_1 = None
        dZ_dev = None
        A_block = None
        Alpha_data_block = None
        self.ctx.Alpha_data_blocks = None

        # da_l = M^T @ dl, da_r = M^T @ dr
        da_l = self.ctx.M_tile_1.T @ cp.asnumpy(dl_tile_1)
        self.bcast_comm.Allreduce(MPI.IN_PLACE, da_l, op=MPI.SUM)
        da_r = self.ctx.M_tile_2.T @ cp.asnumpy(dr_tile_2)
        self.reduce_comm.Allreduce(MPI.IN_PLACE, da_r, op=MPI.SUM)

        self.parameters.a_l.accumulate_grad(da_l)
        self.parameters.a_r.accumulate_grad(da_r)

        # free memory
        self.ctx.M_tile_1 = None
        self.ctx.M_tile_2 = None
        dl_tile_1 = None
        dr_tile_2 = None

        dH_tile_1 = None
        dM_tile_1_dev = cp.asarray(dM_tile_1)
        if not self.is_first_layer:
            # dH = dM @ W^T
            dH_tile_1 = np.zeros_like(self.ctx.H_tile_1, dtype=grad_out.dtype)
            dH_tile_1[0:self.A_shape[0], :] = cp.asnumpy(dM_tile_1_dev[0:self.A_shape[0], :] @ cp.asarray(self.W.T))

        # free memory
        dM_tile_1 = None

        # dW = H^T @ dM
        dW_tile_1 = cp.asnumpy(
            cp.asarray(self.ctx.H_tile_1[0:self.A_shape[0], :].T) @ dM_tile_1_dev[0:self.A_shape[0], :])
        self.bcast_comm.Allreduce(MPI.IN_PLACE, dW_tile_1, op=MPI.SUM)
        self.parameters.W.accumulate_grad(dW_tile_1)

        # free memory
        dM_tile_1_dev = None
        self.ctx.H_tile_1 = None

        return dH_tile_1


class GatModelDistr(gnn_model.GnnModel):
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
            GATconvDistr(channels[i], channels[i + 1], A_shape, ceil_A_shape, tau, use_gpu, bcast_comm, reduce_comm,
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

    parser = argparse.ArgumentParser(description='GAT_distr')
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
        H_tile_1 = utils.generate_dense_matrix(lNK, lNJ, dtype, rng)
    else:
        H_tile_1 = np.empty((lNK, lNJ), dtype=dtype)
    # reduce_comm.Bcast(H_tile_1, root=x)
    utils.bcast_matrix(H_tile_1, reduce_comm, x)

    # The W, a_l, a_r matrices are replicated in all ranks.
    # Therefore, we generate random blocks in cart-rank 0 and then bcast.
    utils.mpi_print(cart_rank, f"Generating weight matrices W with shape ({NJ}, {NL})...")

    if cart_rank == 0:
        W_local = utils.generate_dense_matrix(NJ, NL, dtype, rng) - 0.5
        a_l_local = utils.generate_dense_matrix(NJ, 1, dtype, rng).squeeze() - 0.5
        a_r_local = utils.generate_dense_matrix(NJ, 1, dtype, rng).squeeze() - 0.5
    else:
        W_local = np.empty((NJ, NL), dtype=dtype)
        a_l_local = np.empty(NJ, dtype=dtype)
        a_r_local = np.empty(NJ, dtype=dtype)
    cart_comm.Bcast(W_local, root=0)
    cart_comm.Bcast(a_l_local, root=0)
    cart_comm.Bcast(a_r_local, root=0)

    utils.mpi_print(cart_rank, "Generating adjacency matrix blocks...")
    tau, A_blocks, mappings = gat_model.generate_blocks_training(lA, NJ)
    utils.mpi_print(cart_rank, f"Tile size: {tau} (rows)")

    utils.mpi_print(cart_rank, "Computing forward cpu reference...")
    if cart_rank == 0:
        H_global_ref = np.zeros((bcast_comm.size * lNK, NL), dtype=dtype)
    else:
        H_global_ref = None
    if y == 0:
        bcast_comm.Gather(H_tile_1, H_global_ref, root=0)
    if cart_rank == 0:
        GAT_cpu = gat_model.GATconv(NJ, NL, A.shape, tau, False)
        GAT_cpu.cache_data = True
        GAT_cpu.force_set_parameters(cache_grad=True, W=W_local, a_l=a_l_local, a_r=a_r_local)
        ref_out = GAT_cpu.forward(A, H_global_ref)

        dist_out = np.zeros_like(ref_out)
    else:
        ref_out = None
        dist_out = None

    GAT_dist_gpu = GATconvDistr(NJ, NL, lA.shape, lNI, tau, True, bcast_comm, reduce_comm, cart_comm)
    GAT_dist_gpu.cache_data = True
    GAT_dist_gpu.force_set_parameters(cache_grad=True, W=W_local, a_l=a_l_local, a_r=a_r_local)
    utils.mpi_print(cart_rank, "Computing distributed forward gpu...")
    local_out = GAT_dist_gpu.forward((A_blocks, mappings), H_tile_1)
    if y == 0:
        bcast_comm.Gather(local_out, dist_out, root=0)
    if cart_rank == 0:
        assert np.allclose(ref_out, dist_out, atol=1e-3, rtol=1e-3)
        utils.mpi_print(cart_rank, "Correct distributed forward output!")

    utils.mpi_print(cart_rank, "Computing backward cpu reference...")
    if cart_rank == 0:
        ref_grad = GAT_cpu.backward(A, H_global_ref.copy())
        dist_grad = np.zeros_like(ref_grad)
    else:
        dist_grad = None
        ref_grad = None

    utils.mpi_print(cart_rank, "Computing distributed backward gpu...")
    local_grad = GAT_dist_gpu.backward((A_blocks, mappings), H_tile_1.copy())
    if y == 0:
        bcast_comm.Gather(local_grad, dist_grad, root=0)
    if cart_rank == 0:
        assert np.allclose(ref_grad, dist_grad, atol=1e-3, rtol=1e-3)
        utils.mpi_print(cart_rank, "Correct distributed backward output!")
