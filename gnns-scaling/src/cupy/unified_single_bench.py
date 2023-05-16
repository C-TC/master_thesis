import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp

import gnn_model
import va_model
import gat_model
import agnn_model
import utils

from timeit import repeat

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified single node benchmark.')
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
    parser.add_argument('--layers', type=int, nargs="?", default=3, help='The number of layers in the GNN model.')
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
        kill_signal = np.zeros(1, dtype=np.int32)
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
        # Already create the Kronecker graph distributed
        print(
            f"Generating adjacency matrix for a Kronecker graph with {args['vertices']} vertices and {args['edges']} edges..."
        )
        args['vertices'], args['edges'], A = utils.create_kronecker_graph(args['vertices'], args['edges'], dtype, rng,
                                                                          True)
        print(f"Generated adjacency matrix of Kronecker graph {args['vertices']} vertices and {args['edges']} edges.")

    NI = NK = args['vertices']
    NJ = NL = args['features']
    NNZ = A.nnz

    print(f"Generating feature matrix H with shape ({NK}, {NJ})...")
    target = utils.generate_dense_matrix(NK, NJ, dtype, rng) - 0.5
    H = utils.generate_dense_matrix(NK, NJ, dtype, rng) - 0.5
    grad_out = utils.generate_dense_matrix(NK, NJ, dtype, rng) - 0.5
    print(f"Generating weight matrix W with shape ({NJ}, {NL})...")
    W = utils.generate_dense_matrix(NJ, NL, dtype, rng) - 0.5

    a_l = utils.generate_dense_matrix(NJ, 1, dtype, rng).squeeze() - 0.5
    a_r = utils.generate_dense_matrix(NJ, 1, dtype, rng).squeeze() - 0.5

    print("Generating adjacency matrix blocks...")
    if model_name == 'VA':
        tau, A_blocks, mappings = va_model.generate_blocks_inference(
            A, NJ) if inference_only else va_model.generate_blocks_training(A, NJ)
    elif model_name == 'GAT':
        tau, A_blocks, mappings = gat_model.generate_blocks_inference(
            A, NJ) if inference_only else gat_model.generate_blocks_training(A, NJ)
    else:
        tau, A_blocks, mappings = agnn_model.generate_blocks_inference(
            A, NJ) if inference_only else agnn_model.generate_blocks_training(A, NJ)
    print(f"Tile size: {tau} (rows)")

    # Create the GAT model
    task = 'inference' if inference_only else 'training'
    print(f'Testing {num_layers}-layer {model_name} single node {task} on single process...')

    A_shape = A.shape
    del A

    if model_name == 'GAT':
        model_call_name = 'gat_model.GatModel'
        model_params = 'W=W, a_l=a_l, a_r=a_r'
    elif model_name == 'AGNN':
        model_call_name = 'agnn_model.AGNNmodel'
        model_params = 'W=W'
    else:
        model_call_name = 'va_model.VAmodel'
        model_params = 'W=W'

    model_setup = f"""
model = {model_call_name}([NJ,]*num_layers, NL, A_shape, tau, True, num_layers, inference_only);
for layer in model.layers:
    layer.force_set_parameters(not inference_only, {model_params})
loss = gnn_model.Loss(model, (A_blocks,mappings));
optimizer = gnn_model.Optimizer(model, 0.001);
"""

    # Run the model
    print("Benchmarking the model on GPU...")
    if inference_only:
        gpu_stmt = "model.forward((A_blocks,mappings), H); cp.cuda.get_current_stream().synchronize()"
    else:
        gpu_stmt = "loss.backward(model.forward((A_blocks,mappings), H), target); optimizer.step(); cp.cuda.get_current_stream().synchronize()"

    gpu_setup = model_setup + "cp.cuda.get_current_stream().synchronize()"
    gpu_runtimes = repeat(gpu_stmt,
                          setup=gpu_setup,
                          repeat=num_warmup + num_repeats,
                          number=1,
                          globals={
                              **locals(),
                              **globals()
                          })
    print(
        f"GPU: {utils.time_to_ms(np.median(gpu_runtimes[num_warmup:]))} +- {utils.time_to_ms(np.std(gpu_runtimes[num_warmup:]))}"
    )

    # Logging the results
    filename = 'unified_results.csv'

    with open(filename, 'a') as f:
        # modelname, inference/training, num_nodes, dtype, Vertices, Edges, num_layers, feature_dim, time, std.
        f.write(
            f'unified_single_{model_name}\t{task}\t1\t{dtype}\t{num_vertices}\t{num_edges}\t{num_layers}\t{NJ}\t{utils.time_to_ms(np.median(gpu_runtimes[num_warmup:]))}\t{utils.time_to_ms(np.std(gpu_runtimes[num_warmup:]))}\n'
        )
