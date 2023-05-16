import os
import numpy as np
from scipy import sparse

data_folder = 'data/kron/'
files = [f.name for f in os.scandir(data_folder) if f.is_file()]

nptype = np.float32

def run(file: str):
    el = np.load(os.path.join(data_folder, file))
    assert el.shape[1] == 2
    edges_src = el[:, 0].reshape(-1)
    edges_dst = el[:, 1].reshape(-1)
    num_vertices = np.max(el) + 1
    num_edges = edges_src.shape[0]

    spl = file.replace('.npy', '').split('_')
    nsize = spl[1].split('-')[1]
    esize = spl[2].split('-')[1]
    sparsity = spl[3].split('-')[1]

    A = sparse.coo_matrix((np.ones_like(edges_src), (edges_src, edges_dst)), dtype=nptype)

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
    print(f"working on a{nsize}_e{esize}_s{sparsity}")
    for nodes in [1, 2, 4, 8, 16]:
        raw_dir = os.path.join(data_folder, f"n{nodes}_a{nsize}_e{esize}_s{sparsity}")
        os.makedirs(raw_dir, exist_ok=True)
        graph_path = os.path.join(raw_dir, f"reddit_n{nodes}_a{nsize}_e{esize}_s{sparsity}_graph.npz")
        sparse.save_npz(graph_path, A)
        data_path = os.path.join(raw_dir, "reddit_data.npz")
        np.savez(data_path, **data)


if __name__ == "__main__":
    for file in files:
        run(file)