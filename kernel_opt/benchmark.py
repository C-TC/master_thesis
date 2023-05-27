import argparse
import cupy as cp
import numpy as np
import os
import scipy as sp
import utils
from copy import deepcopy
from timeit import repeat
from cupy_kernel import KERNELS_CUPY, forward_lr_E_rowmax_kernel as GAT_f_0_special
from dace_kernel import KERNELS_DACE, DACE_GPU_NORMAL, DACE_GPU_STRIDED
from cupy_shfl import KERNELS_CUPY_SHFL

KERNELS = [
    "VA_f_0",
    # "GAT_f_0",
    "GAT_f_1",
    "GAT_f_2",
    "GAT_b_0",
    "GAT_b_1",
    "AGNN_f_0",
    "AGNN_b_0",
]

def benchmark(A, kernel, vertices, edges, features, dataset, num_repeats, num_warmup, dtype, rng):
    nnz = A.nnz
    M = N = vertices
    K = features
    sizes = {"M": M, "N": N, "K": K, "nnz": nnz}
    if kernel == "VA_f_0":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            A.indices,
            A.indptr,
            utils.generate_dense_matrix(M, K, dtype, rng),
            utils.generate_dense_matrix(N, K, dtype, rng),
        ]
    elif kernel == "GAT_f_0":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            A.indices,
            A.indptr,
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(N, 1, dtype, rng).squeeze(),
            M,
        ]
    elif kernel == "GAT_f_1":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            A.indices,
            A.indptr,
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
        ]
    elif kernel == "GAT_f_2":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
            A.indices,
            A.indptr,
        ]
    elif kernel == "GAT_b_0":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(M, K, dtype, rng),
            utils.generate_dense_matrix(M, K, dtype, rng),
            A.indices,
            A.indptr,
        ]
    elif kernel == "GAT_b_1":
        outputs = [
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(N, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(N, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            A.indices,
            A.indptr,
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
        ]
    elif kernel == "AGNN_f_0":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            A.indices,
            A.indptr,
            utils.generate_dense_matrix(M, K, dtype, rng),
            utils.generate_dense_matrix(N, K, dtype, rng),
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(N, 1, dtype, rng).squeeze(),
        ]
    elif kernel == "AGNN_b_0":
        outputs = [
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(nnz, 1, dtype, rng).squeeze(),
        ]
        inputs = [
            A.indices,
            A.indptr,
            utils.generate_dense_matrix(M, K, dtype, rng),
            utils.generate_dense_matrix(N, K, dtype, rng),
            utils.generate_dense_matrix(M, K, dtype, rng),
            utils.generate_dense_matrix(N, K, dtype, rng),
            utils.generate_dense_matrix(M, 1, dtype, rng).squeeze(),
            utils.generate_dense_matrix(N, 1, dtype, rng).squeeze(),
        ]
    
    outputs = [cp.asarray(output) for output in outputs]
    inputs = [cp.asarray(input) for input in inputs]

    # cupy grid strided
    if kernel in KERNELS_CUPY:
        gpu_setup = """
for output in outputs:
    output[:] = 0
cp.cuda.get_current_stream().synchronize()
"""
        kernel_func = KERNELS_CUPY[kernel]
        gpu_stmt = """kernel_func[min(65535, M), 128](*outputs, *inputs);cp.cuda.get_current_stream().synchronize()"""
        # if kernel == "GAT_f_0":
        #     gpu_stmt = """GAT_f_0_special((min(65535, M),), (128,), (*outputs, *inputs));cp.cuda.get_current_stream().synchronize()"""

        gpu_runtimes = repeat(gpu_stmt,
                            setup=gpu_setup,
                            repeat=num_warmup + num_repeats,
                            number=1,
                            globals={
                                **locals(),
                                **globals()
                            })
        gpu_runtimes = gpu_runtimes[num_warmup:]
        write_to_file(kernel, 'cupy', M, edges, gpu_runtimes, dataset, rng)
    
    if kernel in KERNELS_CUPY_SHFL:

        gpu_setup = """
for output in outputs:
    output[:] = 0
cp.cuda.get_current_stream().synchronize()
"""
        kernel_func = KERNELS_CUPY_SHFL[kernel]
        gpu_stmt = """kernel_func[min(65535, M), 128](*outputs, *inputs);cp.cuda.get_current_stream().synchronize()"""

        gpu_runtimes = repeat(gpu_stmt,
                            setup=gpu_setup,
                            repeat=num_warmup + num_repeats,
                            number=1,
                            globals={
                                **locals(),
                                **globals()
                            })
        gpu_runtimes = gpu_runtimes[num_warmup:]
        write_to_file(kernel, 'cupy_shfl', M, edges, gpu_runtimes, dataset, rng)
    
    if kernel in KERNELS_DACE:
        # normal version
        sdfg = deepcopy(KERNELS_DACE[kernel])
        DACE_GPU_NORMAL[kernel](sdfg)
        csdfg = sdfg.compile()
        
        gpu_setup = """
for output in outputs:
    output[:] = 0
"""
        gpu_stmt = """csdfg(*outputs, *inputs, **sizes)"""

        gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=num_warmup + num_repeats, number=1, globals=locals())
        gpu_runtimes = gpu_runtimes[num_warmup:]
        write_to_file(kernel, 'dace_normal', M, edges, gpu_runtimes, dataset, rng)
    
    if kernel in KERNELS_DACE:
        # grid strided
        sdfg = deepcopy(KERNELS_DACE[kernel])
        DACE_GPU_STRIDED[kernel](sdfg)
        csdfg = sdfg.compile()
        
        gpu_setup = """
for output in outputs:
    output[:] = 0
"""
        gpu_stmt = """csdfg(*outputs, *inputs, **sizes)"""

        gpu_runtimes = repeat(gpu_stmt, setup=gpu_setup, repeat=num_warmup + num_repeats, number=1, globals=locals())
        gpu_runtimes = gpu_runtimes[num_warmup:]
        write_to_file(kernel, 'dace_strided', M, edges, gpu_runtimes, dataset, rng)


def write_to_file(kernel, method, vertices, edges, gpu_runtimes, dataset, rng, filename='results.csv'):
    median = np.median(gpu_runtimes)
    std = np.std(gpu_runtimes)
    # compute the 95% confidence interval by bootstrapping
    lower, upper = bootstrap_median(gpu_runtimes)
    with open(filename, 'a') as f:
        f.write(f'{kernel},{method},{vertices},{edges},{dataset},{median},{std},{lower},{upper}\n')


def bootstrap_median(data, n_bootstrap_samples=1000, alpha=0.05):
    # Number of data points
    data = np.array(data)
    n = len(data)
    
    # Initialize array to store bootstrap medians
    bootstrap_medians = np.empty(n_bootstrap_samples)
    
    # Generate bootstrap samples and compute medians
    for i in range(n_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_medians[i] = np.median(bootstrap_sample)
    
    # Sort bootstrap medians
    sorted_medians = np.sort(bootstrap_medians)
    
    # Compute lower and upper percentiles
    lower_percentile = np.percentile(sorted_medians, 100 * alpha / 2)
    upper_percentile = np.percentile(sorted_medians, 100 * (1 - alpha / 2))
    
    return lower_percentile, upper_percentile


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
                        default=1000,
                        help='The number of vertices in the graph.')
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=40000, help='The number of edges in the graph.')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        nargs="?",
                        default=None,
                        help='The file containing the adjacency matrix.')
    parser.add_argument('--features', type=int, nargs="?", default=16, help='The number of features.')
    parser.add_argument('--repeat', type=int, nargs="?", default=10, help='The number of times to repeat the benchmark.')
    parser.add_argument('--warmup', type=int, nargs="?", default=1, help='The number of warmup runs.')

    args = vars(parser.parse_args())
    benchmark_dataset = args['dataset']
    num_edges = args['edges']
    num_vertices = args['vertices']
    num_features = args['features']
    num_repeats = args['repeat']
    num_warmup = args['warmup']

    dtype = np.float32

    rng = np.random.default_rng(args['seed'])

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

    for kernel in KERNELS:
        benchmark(A, kernel, num_vertices, num_edges, num_features, benchmark_dataset, num_repeats, num_warmup, dtype, rng)
