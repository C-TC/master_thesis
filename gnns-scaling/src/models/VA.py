import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import torch as th
from dgl.distributed import DistDataLoader

from dgl.utils import expand_as_pair
from dgl import DGLGraph
import dgl.function as fn

import tqdm
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from NeighborSampler import NeighborSampler


class VAconv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(VAconv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.fc = nn.Linear(self._in_dst_feats, self._out_feats, bias=False)

        self.reset_papameters()

    def reset_papameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.fc is not None:
            nn.init.xavier_uniform_(self.fc.weight, gain=gain)

    def forward(self, graph: DGLGraph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata["h"] = feat_src
            graph.dstdata["h"] = feat_dst

            graph.apply_edges(fn.u_dot_v("h", "h", "a"))
            graph.update_all(fn.u_mul_e("h", "a", "m"), fn.sum("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]

            return self.fc(h_neigh)


class VA(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_feats = in_feats
            else:
                in_feats = n_hidden
            if i == n_layers - 1:
                out_feats = n_classes
            else:
                out_feats = n_hidden
            self.layers.append(VAconv(in_feats=in_feats, out_feats=out_feats))

    def forward(self, blocks, x):
        h = x
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
            h = F.relu(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the VA model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        """
        nodes = dgl.distributed.node_split(
            np.arange(g.number_of_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        y = dgl.distributed.DistTensor(
            (g.number_of_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )

        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.number_of_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )

            sampler = NeighborSampler(g, [-1], dgl.distributed.sample_neighbors, device)
            print("|V|={}, eval batch size: {}".format(g.number_of_nodes(), batch_size))
            # Create PyTorch DataLoader for constructing blocks
            dataloader = DistDataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False,
            )

            for blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                h = F.relu(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y
