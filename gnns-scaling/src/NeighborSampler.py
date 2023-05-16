import torch as th
import dgl
import numpy as np


def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        g.ndata["features"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["labels"][seeds].to(device)
    return batch_inputs, batch_labels


class NeighborSampler:
    def __init__(self, g, fanouts, sample_neighbors, device, load_feat=True):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.device = device
        self.load_feat = load_feat

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(
                self.g, seeds, fanout, replace=True
            )
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)

        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]
        batch_inputs, batch_labels = load_subtensor(
            self.g, seeds, input_nodes, "cpu", self.load_feat
        )
        if self.load_feat:
            blocks[0].srcdata["features"] = batch_inputs
        blocks[-1].dstdata["labels"] = batch_labels
        return blocks


