import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import gnn_model
import va_model
import va_model_distr
import gat_model
import gat_model_distr
import agnn_model
import agnn_model_distr
import utils

from mpi4py import MPI
from scipy import sparse
from timeit import repeat

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
    taskname = 'Unified distr benchmark.'
    parser = argparse.ArgumentParser(description=taskname)
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
                        default=50000,
                        help='The number of vertices in the graph.')
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=1000000, help='The number of edges in the graph.')
    parser.add_argument('-t',
                        '--type',
                        nargs="?",
                        choices=['float32', 'float64'],
                        default='float32',
                        help='The type of the data.')
    parser.add_argument('-m',
                        '--model',
                        nargs="?",
                        choices=['VA', 'GAT', 'AGNN'],
                        default='VA',
                        help='The model to test. [VA, GAT, AGNN]')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        nargs="?",
                        default=None,
                        help='The file containing the adjacency matrix.')
    parser.add_argument('--features', type=int, nargs="?", default=128, help='The number of features.')
    parser.add_argument('--inference',
                        action='store_true',
                        help='Run inference only (not storing intermediate matrices).')
    parser.add_argument('-l', '--layers', type=int, nargs="?", default=3, help='The number of layers in the GNN model.')
    parser.add_argument('--repeat', type=int, nargs="?", default=4, help='The number of times to repeat the benchmark.')
    parser.add_argument('--warmup', type=int, nargs="?", default=2, help='The number of warmup runs.')

    args = vars(parser.parse_args())
    inference_only = args['inference']
    num_layers = args['layers']
    num_edges = args['edges']
    num_vertices = args['vertices']
    num_repeats = args['repeat']
    num_warmup = args['warmup']
    model_name = args['model']

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
        # Already create the Kronecker graph distributed
        utils.mpi_print(
            cart_rank,
            f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges..."
        )
        args['vertices'], args['edges'], lA = utils.create_kronecker_graph_distributed(
            args['vertices'], args['edges'], Py, Px, dtype, cart_comm, reduce_comm, rng, True)
        utils.mpi_print(
            cart_rank,
            f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    # Global sizes
    if args['dataset'] == 'kronecker':
        NI = NK = args['vertices']
        NNZ = args['edges']
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
    # Warning: Do not change this, INI and INK are same on all processes respectively.
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
                        block.shape = (min(NI, (x + 1) * lNI) - x * lNI, min(NK, (y + 1) * lNK) - y * lNK)
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
        target_tile_1 = utils.generate_dense_matrix(lNK, lNJ, dtype, rng)
    else:
        H_tile_1 = np.empty((lNK, lNJ), dtype=dtype)
        target_tile_1 = np.empty((lNK, lNJ), dtype=dtype)
    utils.bcast_matrix(H_tile_1, reduce_comm, x)
    utils.bcast_matrix(target_tile_1, reduce_comm, x)
    # reduce_comm.Bcast(H_tile_1, root=x)
    # reduce_comm.Bcast(target_tile_1, root=x)

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
    if model_name == 'VA':
        tau, A_blocks, mappings = va_model.generate_blocks_inference(
            lA, NJ) if inference_only else va_model.generate_blocks_training(lA, NJ)
    elif model_name == 'GAT':
        tau, A_blocks, mappings = gat_model.generate_blocks_inference(
            lA, NJ) if inference_only else gat_model.generate_blocks_training(lA, NJ)
    else:
        tau, A_blocks, mappings = agnn_model.generate_blocks_inference(
            lA, NJ) if inference_only else agnn_model.generate_blocks_training(lA, NJ)
    utils.mpi_print(cart_rank, f"Tile size: {tau} (rows)")

    # Create the GNN model
    task = 'inference' if inference_only else 'training'
    utils.mpi_print(cart_rank, f'Testing {num_layers}-layer {model_name} distributed {task} on {Px}x{Py} processes...')

    lA_shape = lA.shape
    del lA

    if model_name == 'GAT':
        model_call_name = 'gat_model_distr.GatModelDistr'        
        model_params = 'W=W_local, a_l=a_l_local, a_r=a_r_local'
    elif model_name == 'AGNN':
        model_call_name = 'agnn_model_distr.AGNNmodelDistr'
        model_params = 'W=W_local'
    else:
        model_call_name = 'va_model_distr.VAmodelDistr'
        model_params = 'W=W_local'

    model_setup = f"""
model = {model_call_name}([NJ,]*num_layers, NL, lA_shape, lNI, tau, True, bcast_comm, reduce_comm, cart_comm, num_layers, inference_only);
for layer in model.layers:
    layer.force_set_parameters(not inference_only, {model_params})
loss = gnn_model.Loss(model, (A_blocks,mappings));
optimizer = gnn_model.Optimizer(model, 0.001);
"""

    # Run the model
    utils.mpi_print(cart_rank, "Benchmarking the model on GPU...")
    if inference_only:
        gpu_stmt = "model.forward((A_blocks,mappings), H_tile_1); cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
    else:

        gpu_stmt = "loss.backward(model.forward((A_blocks,mappings), H_tile_1), target_tile_1); optimizer.step(); cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"

    gpu_setup = model_setup + "cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
    gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=num_warmup+num_repeats, number=1, globals={**locals(), **globals()})
    utils.mpi_print(cart_rank,
                    f"GPU: {utils.time_to_ms(np.median(gpu_runtimes[num_warmup:]))} +- {utils.time_to_ms(np.std(gpu_runtimes[num_warmup:]))}")
    
    # Logging the results
    filename = 'unified_results.csv'

    if cart_rank == 0:
        with open(filename, 'a') as f:
            # modelname, inference/training, num_nodes, dtype, Vertices, Edges, num_layers, feature_dim, time, std.
            f.write(f'unified_distr_{model_name}\t{task}\t{world_size}\t{dtype}\t{num_vertices}\t{num_edges}\t{num_layers}\t{NJ}\t{utils.time_to_ms(np.median(gpu_runtimes[num_warmup:]))}\t{utils.time_to_ms(np.std(gpu_runtimes[num_warmup:]))}\n')
