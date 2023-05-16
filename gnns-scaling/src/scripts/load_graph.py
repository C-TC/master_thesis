import torch as th

from dgl.data import RedditDataset


class OurDataset(RedditDataset):
    def __init__(self, name, raw_dir="data"):
        self._self_loop_str = f"_{name}"
        self._raw_dir = "/".join(raw_dir.split("/")[:-1])
        self._name = name
        self._verbose = True
        self._transform = None
        self._force_reload = False
        self._url = None
        self._hash_key = ()
        self._hash = self._get_hash()

        self.process()

    def has_cache(self):
        return True

def load_ours(name, root):
    # load reddit data
    data = OurDataset(name, raw_dir=root)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_reddit(self_loop=True, root="dataset"):
    # load reddit data
    data = RedditDataset(self_loop=self_loop, raw_dir=root)
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat")
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    in_feats = graph.ndata["features"].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g
    return train_g, val_g, test_g
