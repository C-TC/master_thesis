import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import kernels
import kernels_gat
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
    """
    Forward cuda kernels:
    kernel1: A_block_data, E_block_data, A_indptr, A_indices, l[0:shape[0]], r[0:shape[1]], row_max
    size:   density * tau * tau, density * tau * tau, tau + 1, density * tau * tau, tau, tau, A.shape[0]
    
    kernel2: Alpha_block_data, E_block_data, row_max, row_sum, A_indptr, A_indices
    size:   density * tau * tau, density * tau * tau, A.shape[0], A.shape[0], tau + 1, density * tau * tau
    nbytes: tau^2 * (2 * density * dtype + density * 4) + tau * (4) + 2 * Ashape[0] * dtype + 4

    kernel3: Alpha_block_data, row_sum, A_indptr, A_indices, M[shape[1],:]
    size:   density * tau * tau, A.shape[0], tau + 1, density * tau * tau, tau * feature_dim
    nbytes: tau^2 * (density * dtype + density * 4) + tau * (feature_dim * dtype + 4) + Ashape[0] * dtype + 4


    memory consumption:
        kernel2 >= kernel1 when blocks > 1, but very close
        if density * tau < feature dim, kernel3 >= kernel2 else kernel3 < kernel2
    """
    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    # case 1: kernel 2
    alpha = 2 * density * dtype + density * 4
    beta = 4
    gamma = 2 * A.shape[0] * dtype + 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau1 = int(np.floor((-beta + abs(delta)) / (2 * alpha)))

    #case 2: kernel 3
    alpha = density * dtype + density * 4
    beta = 4 + feature_dim * dtype
    gamma = A.shape[0] * dtype + 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau2 = int(np.floor((-beta + abs(delta)) / (2 * alpha)))

    tau = min(tau1, tau2)
    tau = min(tau, max(A.shape))
    return utils.generate_blocks_from_tau(A, tau, False)


def generate_blocks_training(
        A: sparse.csr_matrix,
        feature_dim: int) -> Tuple[int, List[List[sparse.csr_matrix]], List[List[List[np.ndarray]]]]:
    """
    Backward cuda kernels:
    kernel1: dAlpha_block_data, Alpha_block_data, dZ[0:shape[0],:], M[0:shape[1],:], row_dot, A_indptr, A_indices
    size:   density * tau * tau, density * tau * tau, tau * feature_dim, tau * feature_dim, A.shape[0], tau + 1, density * tau * tau
    nbytes: tau^2 * (2 * density * dtype + density * 4) + tau * (2 * dtype * feature_dim + 4) + Ashape[0] * dtype + 4

    kernel2: Alpha_data_block, dAlpha_data_block, dl, dr, l[0:shape[0]], r[0:shape[1]], row_dot, A_indptr, A_indices

    kernel3: Alpha_data_block, dM[0:shape[1],:], dZ[0:shape[0],:], A_indptr, A_indices

    memory consumption peak: kernel1, more than any kernel in inference
    """

    density = A.nnz / (A.shape[0] * A.shape[1])
    dtype = np.dtype(A.dtype).itemsize
    available_memory = 0.95 * cp.cuda.Device(0).mem_info[0]

    alpha = 2 * density * dtype + density * 4
    beta = 2 * feature_dim * dtype + 4
    gamma = A.shape[0] * dtype + 4 - available_memory
    delta = np.sqrt(beta**2 - 4 * alpha * gamma)
    tau = int(np.floor((-beta + abs(delta)) / (2 * alpha)))

    tau = min(tau, max(A.shape))

    return utils.generate_blocks_from_tau(A, tau, True)


class GATconv(gnn_model.GnnLayer):
    def __init__(self, in_channel: int, out_channel: int, A_shape: np.ndarray, tau: int, use_gpu: bool) -> None:
        super().__init__(in_channel, out_channel, use_gpu)

        self.A_shape = A_shape
        self.tau = tau

        self.timing_data = []

    def init_parameters(self, rng, dtype, cache_grad: bool = True):
        self.parameters.W = Parameter(
            utils.generate_dense_matrix(self.in_channel, self.out_channel, dtype, rng) - 0.5, cache_grad)
        self.parameters.a_l = Parameter(
            utils.generate_dense_matrix(1, self.out_channel, dtype, rng).squeeze() - 0.5, cache_grad)
        self.parameters.a_r = Parameter(
            utils.generate_dense_matrix(1, self.out_channel, dtype, rng).squeeze() - 0.5, cache_grad)

    def forward_cpu(self, A: sparse.csr_matrix, H: np.ndarray):
        """ Forward pass of GAT layer on CPU, only for debugging purposes.
            
            param A: adjacency matrix
            param input: input H matrix 
        """
        A_dim = max(self.A_shape)
        # M = H @ W, l = M @ a_l, r = M @ a_r
        M = np.zeros((A_dim, self.out_channel), dtype=H.dtype)
        M[0:A.shape[0], :] = H[0:A.shape[0], :] @ self.W
        l = M @ self.a_l
        r = M @ self.a_r

        # D = l + r^T
        D_data = np.zeros_like(A.data, H.dtype)
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                D_data[j] = l[i] + r[A.indices[j]]

        # leaky relu
        E_data = np.maximum(D_data, 0.2 * D_data)

        # softmax row-wise
        row_max = np.full(self.A_shape[0], np.NINF, dtype=H.dtype)
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                row_max[i] = max(row_max[i], E_data[j])
        row_sum = np.zeros(self.A_shape[0], dtype=H.dtype)
        Alpha_data = E_data  # shallow copy
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                # exp(x - max(x))
                Alpha_data[j] = np.exp(Alpha_data[j] - row_max[i])
                row_sum[i] += Alpha_data[j]
        eps = np.finfo(row_sum.dtype).eps
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                Alpha_data[j] /= (row_sum[i] + eps)

        # Z = Alpha @ M
        Z = np.zeros_like(H, dtype=H.dtype)
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                Z[i, :] += M[A.indices[j], :] * Alpha_data[j]

        # relu
        output = np.maximum(Z, 0)

        # cache data if needed in backward pass
        if self.cache_data:
            self.ctx.H = H
            self.ctx.M = M
            self.ctx.l = l
            self.ctx.r = r
            # self.ctx.D_data = D_data
            self.ctx.Alpha_data = Alpha_data
            self.ctx.Z = Z

        return output

    def forward_gpu(self, A_mapping_blocks: Tuple, H: np.ndarray):
        """ Forward pass of GAT layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param input: input H matrix 
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        A_dim = max(self.A_shape)
        # M = H @ W
        # TODO: optimize these three line (compute on the fly?)
        M = cp.zeros((A_dim, self.out_channel), dtype=H.dtype)
        M[0:self.A_shape[0], :] = cp.asarray(H[0:self.A_shape[0], :]) @ cp.asarray(self.W)
        l = cp.asnumpy(M @ cp.asarray(self.a_l))
        r = cp.asnumpy(M @ cp.asarray(self.a_r))
        M = cp.asnumpy(M)

        # E_data_blocks similar to A_blocks but only data of E
        E_data_blocks: List[List[np.ndarray]] = []
        # also compute row_max of E
        E_row_max_dev = cp.full(self.A_shape[0], cp.NINF, dtype=H.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            E_data_blocks_i = []
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                E_data_blocks_i.append(
                    self.gat_forward_gpu_kernel_fuse_E_rowmax(i, k, A_block, H, self.W, M, l, r, E_row_max_dev))
            E_data_blocks.append(E_data_blocks_i)

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

        # compute the division in Alpha and Z = Alpha @ M
        alpha_row_sum_dev += cp.finfo(alpha_row_sum_dev.dtype).eps
        Z = np.zeros_like(H, dtype=H.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = Alpha_data_blocks[i // self.tau][k // self.tau]
                self.gat_forward_gpu_kernel_softmax_2_matmul_M(i, k, A_block, Alpha_data_block, M, Z, alpha_row_sum_dev)

        # relu
        # TODO: fuse this in Z = Alpha @ M?
        output = np.maximum(Z, 0)

        if self.cache_data:
            self.ctx.H = H
            self.ctx.M = M
            self.ctx.l = l
            self.ctx.r = r
            self.ctx.Alpha_data_blocks = Alpha_data_blocks
            self.ctx.Z = Z

        return output

    def backward_cpu(self, A: sparse.csr_matrix, grad_out: np.ndarray):
        """ Backward pass of GAT layer on CPU, only for debugging purposes.
            
            param A: adjacency matrix
            param grad_out: gradient of output
        """
        A_dim = max(self.A_shape)
        # relu
        dZ = grad_out * (self.ctx.Z > 0)

        # dAlpha = dZ * M^T
        d_Alpha_data = np.zeros_like(A.data, dtype=grad_out.dtype)
        for row in range(len(A.indptr) - 1):
            for col_idx in range(A.indptr[row], A.indptr[row + 1]):
                col = A.indices[col_idx]
                d_Alpha_data[col_idx] = np.dot(dZ[row, :], self.ctx.M[col, :])

        # dE[i,:] = (dAlpha[i,:] - dAlpha[i,:] @ Alpha[i,:].T) * Alpha[i,:]
        dot_prod = np.zeros(A.shape[0], dtype=grad_out.dtype)
        for i in range(len(A.indptr) - 1):
            dot_prod[i] = np.dot(self.ctx.Alpha_data[A.indptr[i]:A.indptr[i + 1]],
                                 d_Alpha_data[A.indptr[i]:A.indptr[i + 1]])

        dE_data = np.zeros_like(A.data, dtype=grad_out.dtype)
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                dE_data[j] = (d_Alpha_data[j] - dot_prod[i]) * self.ctx.Alpha_data[j]

        # leaky relu
        dD_data = np.zeros_like(A.data, dtype=grad_out.dtype)
        D_data = np.zeros_like(A.data, dtype=grad_out.dtype)
        for i in range(len(A.indptr) - 1):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                D_data[j] = self.ctx.l[i] + self.ctx.r[A.indices[j]]
        dD_data = dE_data * (D_data > 0) + 0.2 * dE_data * (D_data <= 0)

        # dl = sum_row dD, dr = sum_col dD
        dl = np.zeros(A_dim, dtype=grad_out.dtype)
        dr = np.zeros(A_dim, dtype=grad_out.dtype)
        for row in range(len(A.indptr) - 1):
            for col_idx in range(A.indptr[row], A.indptr[row + 1]):
                col = A.indices[col_idx]
                dl[row] += dD_data[col_idx]
                dr[col] += dD_data[col_idx]

        # dM = dl @ a_l.T + dr @ a_r.T + Alpha.T @ dZ
        dM = np.zeros_like(self.ctx.M, dtype=grad_out.dtype)
        dM[0:self.A_shape[0], :] += dl[0:self.A_shape[0], None] @ self.a_l[None, :]
        dM[0:self.A_shape[1], :] += dr[0:self.A_shape[1], None] @ self.a_r[None, :]
        Alpha = sparse.csr_matrix((self.ctx.Alpha_data, A.indices, A.indptr), shape=A.shape)
        dM[0:self.A_shape[1], :] += Alpha.T @ dZ[0:self.A_shape[0], :]

        # da_l = M^T @ dl, da_r = M^T @ dr
        da_l = self.ctx.M.T @ dl
        da_r = self.ctx.M.T @ dr
        self.parameters.a_l.accumulate_grad(da_l)
        self.parameters.a_r.accumulate_grad(da_r)

        dH = None
        if not self.is_first_layer:
            # dH = dM @ W^T
            dH = np.zeros_like(self.ctx.H, dtype=grad_out.dtype)
            dH[0:self.A_shape[0], :] = dM[0:self.A_shape[0], :] @ self.W.T

        # dW = H^T @ dM
        dW = self.ctx.H[0:A.shape[0], :].T @ dM[0:A.shape[0], :]
        self.parameters.W.accumulate_grad(dW)

        return dH

    def backward_gpu(self, A_mapping_blocks: Tuple, grad_out: np.ndarray):
        """ Backward pass of GAT layer on GPU.
            
            param A_mapping_blocks: Tuple of A blocks and mapping blocks
            param grad_out: gradient of output
        """
        A_blocks: List[List[sparse.csr_matrix]] = A_mapping_blocks[0]
        mapping_blocks: List[List[List[np.ndarray]]] = A_mapping_blocks[1]
        A_dim = max(self.A_shape)
        start_time = time.perf_counter()
        cscmm_time = 0.0

        # relu
        dZ = grad_out * (self.ctx.Z > 0)
        # free memory
        self.ctx.Z = None

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
                    self.gat_backward_gpu_kernel_Z_Alpha_rowdot(i, k, A_block, Alpha_data_block, dZ, self.ctx.M,
                                                                dAlpha_dot_Alpha_dev))
            dAlpha_blocks.append(dAlpha_blocks_i)

        # backprop softmax, recompute D, backprop leaky relu, compute dl, dr
        dl = cp.zeros(A_dim, dtype=grad_out.dtype)
        dr = cp.zeros(A_dim, dtype=grad_out.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = self.ctx.Alpha_data_blocks[i // self.tau][k // self.tau]
                dAlpha_block = dAlpha_blocks[i // self.tau][k // self.tau]
                self.gat_backward_gpu_kernel_softmax_leakyrelu_lr(i, k, A_block, Alpha_data_block, dAlpha_block,
                                                                  self.ctx.l, self.ctx.r, dl, dr, dAlpha_dot_Alpha_dev)

        # free memory
        dAlpha_dot_Alpha_dev = None
        dAlpha_blocks = None
        self.ctx.l = None
        self.ctx.r = None

        # dM = dl @ a_l.T + dr @ a_r.T + Alpha.T @ dZ
        time_0 = time.perf_counter()
        dM = np.zeros_like(self.ctx.M, dtype=grad_out.dtype)
        for i in range(0, self.A_shape[0], self.tau):
            dZ_dev = cp.asarray(dZ[i:i + A_blocks[i // self.tau][0].shape[0], :])
            for k in range(0, self.A_shape[1], self.tau):
                A_block = A_blocks[i // self.tau][k // self.tau]
                Alpha_data_block = self.ctx.Alpha_data_blocks[i // self.tau][k // self.tau]
                Alpha_T_dev = cp.sparse.csr_matrix((cp.asarray(Alpha_data_block), cp.asarray(A_block.indices), cp.asarray(A_block.indptr)),
                                                   shape=A_block.shape).T
                dM[k:k + A_block.shape[1], :] += cp.asnumpy(Alpha_T_dev @ dZ_dev)
        cscmm_time += time.perf_counter() - time_0
        dM[0:self.A_shape[0], :] += cp.asnumpy(dl[0:self.A_shape[0], None] @ cp.asarray(self.a_l[None, :]))
        dM[0:self.A_shape[1], :] += cp.asnumpy(dr[0:self.A_shape[1], None] @ cp.asarray(self.a_r[None, :]))

        # free memory
        dZ = None
        dZ_dev = None
        Alpha_dev = None
        A_block = None
        Alpha_data_block = None
        self.ctx.Alpha_data_blocks = None

        # da_l = M^T @ dl, da_r = M^T @ dr
        self.parameters.a_l.accumulate_grad(self.ctx.M.T @ cp.asnumpy(dl))
        self.parameters.a_r.accumulate_grad(self.ctx.M.T @ cp.asnumpy(dr))

        # free memory
        self.ctx.M = None
        dl = None
        dr = None

        dH = None
        dM_dev = cp.asarray(dM)
        if not self.is_first_layer:
            # dH = dM @ W^T
            dH = np.zeros_like(self.ctx.H, dtype=grad_out.dtype)
            dH[0:self.A_shape[0], :] = cp.asnumpy(dM_dev[0:self.A_shape[0], :] @ cp.asarray(self.W.T))

        # free memory
        dM = None

        # dW = H^T @ dM
        self.parameters.W.accumulate_grad(
            cp.asnumpy(cp.asarray(self.ctx.H[0:self.A_shape[0], :].T) @ dM_dev[0:self.A_shape[0], :]))

        # free memory
        dM_dev = None
        self.ctx.H = None

        end_time = time.perf_counter()
        self.timing_data.extend([cscmm_time, end_time - start_time])

        return dH

    def gat_forward_gpu_kernel_fuse_E_rowmax(self, row_start: int, col_start: int, A_block: sparse.csr_matrix,
                                             H: np.ndarray, W: np.ndarray, M: np.ndarray, l: np.ndarray, r: np.ndarray,
                                             row_max_dev: cp.ndarray) -> np.ndarray:
        """ First part of forward pass of GAT on gpu, fuse computation from l,r to E. 
            TODO: fuse computation from begin to E.
            Also compute row-wise max of E and store in row_max.
                
            param row_start: start row index of current block of A in A
            param col_start: start column index of current block of A in A
            param A_block: current block of adjacency matrix
            param H: H on host memory
            param W: W on host memory
            param M: preallocated? M on host memory
            param l: preallocated? l on host memory
            param r: preallocated? r on host memory
            row_max_dev: preallocated? row_max on device memory
            return: block of E on host memory
        """
        E_data = cp.zeros_like(A_block.data, row_max_dev.dtype)
        kernels_gat.forward_lr_E_rowmax_kernel(
            (min(65535,
                 len(A_block.indptr) - 1), ), (128, ),
            (E_data, row_max_dev[row_start:row_start + A_block.shape[0]], cp.asarray(
                A_block.indices), cp.asarray(A_block.indptr), cp.asarray(l[row_start:row_start + A_block.shape[0]]),
             cp.asarray(r[col_start:col_start + A_block.shape[1]]), len(A_block.indptr) - 1))
        return cp.asnumpy(E_data)

    def gat_forward_gpu_kernel_softmax_1(self, row_start: int, col_start: int, A_block: sparse.csr_matrix,
                                         E_data_block: np.ndarray, row_max_dev: cp.ndarray,
                                         row_sum_dev: cp.ndarray) -> np.ndarray:
        """ Second part of forward pass of GAT on gpu.
            Compute tmp = exp(E - row_max) and row_sum = sum(tmp of this row)
            :param row_start: start row index of current block of A in A
            :param col_start: start column index of current block of A in A
            :param A_block: current block of adjacency matrix
            :param E_data_block: current block of E_data
            :param row_max_dev: row_max on device memory
            :param row_sum_dev: preallocated row_sum on device memory
        """
        Alpha_data = cp.zeros_like(E_data_block, E_data_block.dtype)
        kernels_gat.gat_forward_softmax_1[min(65535,
                                              len(A_block.indptr) - 1),
                                          128](Alpha_data, row_sum_dev[row_start:row_start + A_block.shape[0]],
                                               cp.asarray(E_data_block), cp.asarray(A_block.indices),
                                               cp.asarray(A_block.indptr),
                                               row_max_dev[row_start:row_start + A_block.shape[0]])
        return cp.asnumpy(Alpha_data)

    def gat_forward_gpu_kernel_softmax_2_matmul_M(self, row_start: int, col_start: int, A_block: sparse.csr_matrix,
                                                  Alpha_data_block: np.ndarray, M: np.ndarray, Z: np.ndarray,
                                                  row_sum_dev: cp.ndarray) -> None:
        """ Third part of forward pass of GAT on gpu.
            Compute Alpha = tmp_Alpha / row_sum and Z = Alpha @ M 
            :param row_start: start row index of current block of A in A
            :param col_start: start column index of current block of A in A
            :param A_block: current block of adjacency matrix
            :param Alpha_data_block: current block of Alpha_data (read and write back)
            :param M: M on host memory
            :param Z: preallocated Z on host memory
            :param row_sum_dev: row_sum on device memory
        """
        Alpha_data_dev = cp.asarray(Alpha_data_block)
        indices_dev = cp.asarray(A_block.indices)
        indptr_dev = cp.asarray(A_block.indptr)
        kernels_gat.gat_forward_softmax_2[min(65535,
                                              len(A_block.indptr) - 1),
                                          128](Alpha_data_dev, row_sum_dev[row_start:row_start + A_block.shape[0]],
                                               indices_dev, indptr_dev)
        Alpha_dev = cp.sparse.csr_matrix((Alpha_data_dev, indices_dev, indptr_dev), shape=A_block.shape)
        Alpha_data_block[:] = cp.asnumpy(Alpha_data_dev)
        Z[row_start:row_start + A_block.shape[0], :] += cp.asnumpy(
            Alpha_dev @ cp.asarray(M[col_start:col_start + A_block.shape[1], :]))

    def gat_backward_gpu_kernel_Z_Alpha_rowdot(self, row_start: int, col_start: int, A_block: sparse.csr_matrix,
                                               Alpha_data_block: np.ndarray, dZ: np.ndarray, M: np.ndarray,
                                               row_dot_dev: cp.ndarray) -> np.ndarray:
        """ First part of backward pass of GAT on gpu.
            Compute dAlpha = A * (dZ @ M.T) and row_dot(i) = dAlpha(i,:) @ Alpha(i,:).T
            :param row_start: start row index of current block of A in A
            :param col_start: start column index of current block of A in A
            :param A_block: current block of adjacency matrix
            :param Alpha_data_block: current block of Alpha_data
            :param dZ: dZ on host memory
            :param M: M on host memory
            :param row_dot_dev: preallocated row_dot on device memory
            :return: current block of dAlpha_data
        """
        dAlpha_data = cp.zeros_like(Alpha_data_block, Alpha_data_block.dtype)
        kernels_gat.gat_backward_dZ_dAlpha_rowdot_shfl[min(65535,
                                                      len(A_block.indptr) - 1),
                                                  128](dAlpha_data, row_dot_dev[row_start:row_start + A_block.shape[0]],
                                                       cp.asarray(Alpha_data_block),
                                                       cp.asarray(dZ[row_start:row_start + A_block.shape[0], :]),
                                                       cp.asarray(M[col_start:col_start + A_block.shape[1], :]),
                                                       cp.asarray(A_block.indices), cp.asarray(A_block.indptr))
        return cp.asnumpy(dAlpha_data)

    def gat_backward_gpu_kernel_softmax_leakyrelu_lr(self, row_start: int, col_start: int, A_block: sparse.csr_matrix,
                                                     Alpha_data_block: np.ndarray, dAlpha_data_block: np.ndarray,
                                                     l: np.ndarray, r: np.ndarray, dl: cp.ndarray, dr: cp.ndarray,
                                                     row_dot_dev: cp.ndarray) -> None:
        """ Second part of backward pass of GAT on gpu.
            Compute dE(i,:) = (dAlpha(i,:) - row_dot(i)) * Alpha(i,:)
            Recompute D(i,j) = l(i) + r(j)
            Compute dD from E = leakyrelu(D)
            Compute dl(i) = sum(dD(i,:)), dr(j) = sum(dD(:,j))
            :param row_start: start row index of current block of A in A
            :param col_start: start column index of current block of A in A
            :param A_block: current block of adjacency matrix
            :param Alpha_data_block: current block of Alpha_data
            :param dAlpha_data_block: current block of dAlpha_data
            :param l: l on host memory
            :param r: r on host memory
            :param dl: preallocated dl on device memory
            :param dr: preallocated dr on device memory
            :param row_dot_dev: row_dot on device memory
        """
        kernels_gat.gat_backward_softmax_leakyrelu_lr[min(65535,
                                                          len(A_block.indptr) - 1),
                                                      128](dl[row_start:row_start + A_block.shape[0]],
                                                           dr[col_start:col_start + A_block.shape[1]],
                                                           cp.asarray(l[row_start:row_start + A_block.shape[0]]),
                                                           cp.asarray(r[col_start:col_start + A_block.shape[1]]),
                                                           cp.asarray(Alpha_data_block), cp.asarray(dAlpha_data_block),
                                                           cp.asarray(A_block.indices), cp.asarray(A_block.indptr),
                                                           row_dot_dev[row_start:row_start + A_block.shape[0]])


class GatModel(gnn_model.GnnModel):
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 A_shape: List[int],
                 tau: int,
                 use_gpu: bool,
                 num_layers: int,
                 inference_only=False) -> None:
        """ Initialize GAT model.
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
        layers = [GATconv(channels[i], channels[i + 1], A_shape, tau, use_gpu) for i in range(num_layers)]
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

    parser = argparse.ArgumentParser(description='GAT single node')
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
    print(f"Generating weight arrays a_l, a_r with shape ({NJ})...")

    print("Generating adjacency matrix blocks...")
    tau, A_blocks, mapping_blocks = generate_blocks_training(A, NJ)
    print(f"Tile size: {tau} (rows)")

    print("Computing forward reference (CPU) output...")
    GAT_cpu = GATconv(NJ, NL, A.shape, tau, False)
    GAT_cpu.cache_data = True
    GAT_cpu.init_parameters(rng, dtype, True)
    ref_out = GAT_cpu.forward(A, H)

    print("Computing GPU output...")
    GAT_gpu = GATconv(NJ, NL, A.shape, tau, True)
    GAT_gpu.cache_data = True
    GAT_gpu.parameters = deepcopy(GAT_cpu.parameters)
    gpu_out = GAT_gpu.forward((A_blocks, mapping_blocks), H)
    print("Validating results...")
    assert cp.allclose(ref_out, gpu_out, rtol=1e-3, atol=1e-3)
    print("Forward pass validation passed.")

    print("Computing backward reference (CPU) output...")
    ref_grad = GAT_cpu.backward(A, grad_out)

    print("Computing GPU output...")
    gpu_grad = GAT_gpu.backward((A_blocks, mapping_blocks), grad_out)
    print("Validating results...")
    assert cp.allclose(ref_grad, gpu_grad, rtol=1e-3, atol=1e-3)
    print("Backward pass validation passed.")
