import os
import argparse
import numpy as np

from scipy import sparse


grid = {
    #  [Px, Py]
    1: [1, 1],
    2: [1, 2],
    4: [2, 2],
    8: [2, 4],
    16: [4, 4],
    32: [4, 8],
    64: [8, 8],
    128: [8, 16],
    256: [16, 16],
    512: [16, 32],
}


nptype = np.float32
scalf = np.sqrt(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-as", "--a-size", type=int, nargs="?", default=17)
    parser.add_argument("-hc", "--h-cols", type=int, nargs="?", default=128)
    parser.add_argument("-sp", "--sparsity", type=float, nargs="?", default=1e-2)
    parser.add_argument("-nn", "--num-nodes", type=int, nargs="?", default=1)
    parser.add_argument("-o", "--output", type=str, nargs="?", default="data")

    args = vars(parser.parse_args())

    print(f"Script called with options: {args}")

    nodes = args["num_nodes"]
    assert nodes in grid

    rng = np.random.default_rng(42)

    Nx, Ny = grid[nodes]
    NArows = int(np.ceil(2**args["a_size"] * np.sqrt(nodes) / nodes)) * nodes
    LArows = NArows // Nx
    LAcols = NArows // Ny
    NHcols = NWcols = args["h_cols"]
    density = args["sparsity"]

    raw_dir = os.path.join(args["output"], f"n{nodes}_a{NArows}_s{density}")
    os.makedirs(raw_dir, exist_ok=True)

    lA = sparse.random(
        LArows, LAcols, density=density, format="csr", dtype=nptype, random_state=rng
    )
    lA.data[:] = 1
    A = sparse.bmat([[lA] * Ny] * Nx, format="coo", dtype=nptype)

    graph_path = os.path.join(raw_dir, f"reddit_n{nodes}_a{NArows}_s{density}_graph.npz")
    sparse.save_npz(graph_path, A)

    nnz = len(A.data)
    lH = rng.random((LArows, NHcols), dtype=nptype)
    H = np.repeat(lH, Nx, axis=0)
    assert H.shape == (NArows, NHcols)
    data = {
        "feature": H,
        "node_types": rng.integers(1, 4, size=NArows, dtype=np.int32),
        "node_ids": np.arange(NArows, dtype=np.int32),
        "label": rng.integers(0, 128, size=NArows, dtype=np.int32),
    }
    data_path = os.path.join(raw_dir, "reddit_data.npz")
    np.savez(data_path, **data)
