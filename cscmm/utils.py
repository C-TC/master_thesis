import ctypes
import cupy as cp
import numpy as np
import os
import math
import scipy as sp

from scipy import sparse
from timeit import repeat
from typing import List, Tuple


def generate_sparse_matrix(rows: int, cols: int, nnz: int, dtype: np.dtype,
                           rng: np.random.Generator) -> sparse.csr_matrix:
    """ Generates a sparse matrix in CSR format with the given number of non-zero elements.

    NOTE: The number of non-zero elements may be slightly larger than the requested number due to rounding.

    :param rows: The number of rows in the matrix.
    :param cols: The number of columns in the matrix.
    :param nnz: The number of non-zero elements in the matrix.
    :param dtype: The data type of the matrix.
    :param rng: The random number generator to use.
    :return: The generated sparse matrix.
    """
    # density = nnz / (rows * cols)
    # return sparse.random(rows, cols, density=density, format='csr', dtype=dtype, random_state=rng)
    # NOTE: The following avoids issues with sparse.random failing due to overflowing indices
    nnzpr = int(np.ceil(nnz / rows))
    actual_nnz = nnzpr * rows
    data = rng.random((actual_nnz, ), dtype=dtype)
    indptr = np.arange(0, actual_nnz + 1, nnzpr, dtype=np.int64)
    indices = rng.integers(0, cols, size=(actual_nnz, ), dtype=np.int64)
    tmp = sparse.csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=dtype)
    tmp.sum_duplicates()
    tmp.sort_indices()
    return tmp


def generate_dense_matrix(rows: int, cols: int, dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    """ Generates a dense matrix.

    :param rows: The number of rows in the matrix.
    :param cols: The number of columns in the matrix.
    :param dtype: The data type of the matrix.
    :param rng: The random number generator to use.
    :return: The generated dense matrix.
    """
    return rng.random((rows, cols), dtype=dtype)


def load_adjacency_matrix_csr(folder: str,
                              filename: str,
                              suffix: str = "npy",
                              row_idx: int = 1,
                              num_rows: int = None,
                              dtype: np.dtype = np.float32) -> sparse.csr_matrix:
    """ Loads an adjacency matrix in COO format from file and returns it as a CSR matrix.
    
    The following assumptions are made:
    - The file is in uncompressed numpy format (.npy)
    - The file contains a 2D array of shape (2, num_edges)

    :param folder: The folder where the file is located.
    :param filename: The name of the file.
    :param suffix: The suffix of the file.
    :param row_idx: The index of the rows in the file.
    :param num_rows: The number of rows in the matrix.
    :param dtype: The data type of the matrix.
    :return: The adjacency matrix as a CSR matrix.
    """
    coo_indices = np.load(os.path.join(folder, f"{filename}.{suffix}"))
    rows = coo_indices[row_idx]
    cols = coo_indices[1 - row_idx]
    data = np.ones(len(rows), dtype=dtype)

    num_rows = num_rows or rows.max() + 1
    if num_rows < rows.max() + 1:
        raise ValueError("The number of rows in the file is larger than the specified number of rows.")
    tmp = sparse.csr_matrix((data, (rows, cols)), shape=(num_rows, num_rows), dtype=dtype)
    tmp.sum_duplicates()
    tmp.sort_indices()
    return tmp


def time_to_ms(runtime: float) -> int:
    return int(runtime * 1000)


# function create_kronecker_graph was written by Patrick Iff
#
# Please build the necessary shared library first, before calling this function. This can be done by
# calling make in the kronecker directory.
#
# Arguments
# - vertex_count:       Number of vertices that the Kronecker graph should have
# - target_edge_count:  Number of edges that the Kronecker graph should have. Note that the real number
# - target_data_type:   The NumPy datatype that should be used to stores 0s and 1s in the CSR matrix
# - rng:                Randon number generator
# - fix_empty_rows:     If set to true, the function inserts a random 1 into each row that does not contain non-zeros
#
# Return value:
# - A tuple of the form (n, m, mat) where
# -- n is the number of vertices in the generated Kronecker graph
# -- m is the number of edges in the generated Kronecker graph
# -- mat is the adjacency matrix of the generated Kronecker graph in CSR format

def create_kronecker_graph(vertex_count: int, target_edge_count: int, target_data_type: np.dtype,
                           rng: np.random.Generator, fix_empty_rows = False):
    # Check arguments: The vertex count must be a power of two
    if math.log2(vertex_count) % 1 != 0:
        invalid_vertex_count = vertex_count
        vertex_count = int(2**(math.floor(math.log2(vertex_count))))
        print("WARNING: The vertex count must be a power of two. " + \
              "An invalid vertex count of %d was given which was adjusted to %d" \
              % (invalid_vertex_count, vertex_count))

    # Compute parameters needed by the python function
    scale = int(math.log2(vertex_count))
    # We round up since duplicated edges will be removed and hence, the actual edge count usually
    # is below the target edge count, hence, rounding up can't be wrong.
    edge_factor = int(math.ceil(target_edge_count / vertex_count))

    # The C function that creates the Kronecker graph will store the number of actual edges and
    # pointers to the lists of rows and columns into these three variables
    output_edge_count = (ctypes.c_ulonglong)()
    # Pointer to an array of c_uint32, but we assume 64-bit addresses, hence, the pointer is of type c_ulonglong.
    output_edge_rows = (ctypes.c_ulonglong)()
    output_edge_cols = (ctypes.c_ulonglong)()

    graph_lib = ctypes.CDLL("./kronecker/graph.so")

    # Call C function to create the Kronecker graph
    graph_lib.createEdgesSingleNode(    scale, \
                                        edge_factor, \
                                        ctypes.byref(output_edge_count), \
                                        ctypes.byref(output_edge_rows), \
                                        ctypes.byref(output_edge_cols))

    # Convert edge list into two lists
    real_edge_count = output_edge_count.value
    # The additional lists are not ideal memory-wise, but otherwise we won't be able to add additional edges
    rows = list(np.ctypeslib.as_array((ctypes.c_uint32 * real_edge_count).from_address(output_edge_rows.value)))
    cols = list(np.ctypeslib.as_array((ctypes.c_uint32 * real_edge_count).from_address(output_edge_cols.value)))

    # Call C function to free the edge list from memory
    graph_lib.freeEdges(ctypes.byref(output_edge_rows), ctypes.byref(output_edge_cols))

    # For each row without any non-zeros (i.e. for each vertex without any outgoing edges)
    # Insert a 1 in a random column, s.t. it is not on the diagonal (i.e. the outgoing edge is no self-loop).
    if fix_empty_rows:
        rows_present = np.zeros(vertex_count, dtype=np.uint8)
        for i in rows:
            rows_present[i] = 1
        for i in range(vertex_count):
            if rows_present[i] == 0:
                random_col = rng.integers(0, vertex_count-2, size=1, dtype=np.int64)
                random_col[0] += (1 if random_col[0] >= i else 0)
                rows.append(i)
                cols.append(random_col[0])
                real_edge_count += 1

    # Convert the edge list to a SciPy CSR matrix
    vals = np.ones(real_edge_count, dtype=target_data_type)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(vertex_count, vertex_count), dtype=target_data_type)

    # Replace multi-edges by single-edges
    mat.data = np.ones(len(mat.data), dtype=target_data_type)

    mat.sort_indices()

    return (vertex_count, mat.nnz, mat)


def sp2cp(matrix: sparse.csr_matrix) -> cp.sparse.csr_matrix:
    """ Converts a SciPy CSR matrix to a CuPy CSR matrix. 
    
    :param matrix: The SciPy CSR matrix.
    :return: The CuPy CSR matrix.
    """
    tmp = cp.sparse.csr_matrix((cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
                               shape=matrix.shape,
                               dtype=matrix.dtype)
    tmp._has_canonical_format = True
    return tmp



def generate_blocks_from_tau(
        A: sparse.csr_matrix, tau: int,
        generate_mapping: bool) -> Tuple[int, List[List[sparse.csr_matrix]], List[List[List[np.ndarray]]]]:
    A_blocks = []
    mapping_blocks = []
    if tau >= A.shape[0] and tau >= A.shape[1]:
        A_blocks.append([
            A,
        ])
        if generate_mapping:
            mapping = sparse.csc_matrix((np.arange(A.data.shape[0]), A.indices, A.indptr),
                                        shape=(A.shape[1], A.shape[0])).tocsr(copy=True)
            mapping.sum_duplicates()
            mapping.sort_indices()
            mapping_blocks.append([
                (np.array(mapping.data, dtype=np.int32), mapping.indices, mapping.indptr),
            ])
    else:
        for i in range(0, A.shape[0], tau):
            A_blocks_i = []
            mapping_blocks_i = []

            for k in range(0, A.shape[1], tau):
                tmp = sparse.csr_matrix(A[i:min(i + tau, A.shape[0]), k:min(k + tau, A.shape[1])])
                tmp.sum_duplicates()
                tmp.sort_indices()
                A_blocks_i.append(tmp)
                if generate_mapping:
                    mapping = sparse.csc_matrix((np.arange(tmp.data.shape[0]), tmp.indices, tmp.indptr),
                                                shape=(tmp.shape[1], tmp.shape[0])).tocsr(copy=True)
                    mapping.sum_duplicates()
                    mapping.sort_indices()
                    mapping_blocks_i.append((np.array(mapping.data, dtype=np.int32), mapping.indices, mapping.indptr))

            A_blocks.append(A_blocks_i)
            mapping_blocks.append(mapping_blocks_i)

    return tau, A_blocks, mapping_blocks
