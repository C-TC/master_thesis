import ctypes
import os
import math
import numpy as np
from scipy import sparse

data_folder = 'data/kron/'

nptype = np.float32

if __name__ == "__main__":
    vertex_count = 16
    target_edge_count = 32
    snnz = "0.0001"

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

    graph_lib = ctypes.CDLL("./graph.so")

    # Call C function to create the Kronecker graph
    graph_lib.createEdgesSingleNode(    scale, \
                                        edge_factor, \
                                        ctypes.byref(output_edge_count), \
                                        ctypes.byref(output_edge_rows), \
                                        ctypes.byref(output_edge_cols))

    # Convert edge list into two lists
    edge_count = output_edge_count.value

    edges_src = np.ctypeslib.as_array((ctypes.c_uint32 * edge_count).from_address(output_edge_rows.value))
    edges_dst = np.ctypeslib.as_array((ctypes.c_uint32 * edge_count).from_address(output_edge_cols.value))
    num_vertices = vertex_count
    num_edges = edges_src.shape[0]

    A = sparse.coo_matrix((np.ones_like(edges_src), (edges_src, edges_dst)), shape=(vertex_count, vertex_count), dtype=nptype).todok().tocoo()

    # Call C function to free the edge list from memory
    graph_lib.freeEdges(ctypes.byref(output_edge_rows), ctypes.byref(output_edge_cols))

    rng = np.random.default_rng(42)

    H = rng.random((num_vertices, 128), dtype=nptype)
    ntypes = rng.integers(1, 4, size=num_vertices, dtype=np.int32)
    nids = np.arange(num_vertices, dtype=np.int32)
    lab = rng.integers(0, 128, size=num_vertices, dtype=np.int32)

    data = {
        "feature": H,
        "node_types": ntypes,
        "node_ids": nids,
        "label": lab,
    }

    for nodes in [1, 2, 4, 8, 16]:
        raw_dir = os.path.join(data_folder, f"n{nodes}_a{scale}_e{target_edge_count}_s{snnz}")
        os.makedirs(raw_dir, exist_ok=True)
        graph_path = os.path.join(raw_dir, f"reddit_n{nodes}_a{scale}_e{target_edge_count}_s{snnz}_graph.npz")
        sparse.save_npz(graph_path, A)
        data_path = os.path.join(raw_dir, "reddit_data.npz")
        np.savez(data_path, **data)
