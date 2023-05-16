import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

import dgl
import numpy as np
from dgl.distributed import DistDataLoader

import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair

import tqdm
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from NeighborSampler import NeighborSampler


def udf_mean(nodes):
    return {'h': th.mean(nodes.mailbox['m'], dim=1)}


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 norm='none',
                 allow_zero_in_degree=True):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
            init.xavier_uniform_(self.weight)
        else:
            self.register_parameter('weight', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_u('h', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum("m", "h"))
            rst = graph.dstdata['h']
            if self.weight is not None:
                rst = th.matmul(rst, self.weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            return rst


class GCN(nn.Module):
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
            self.layers.append(
                GraphConv(in_feats=in_feats, out_feats=out_feats)
            )

        self.dropout = nn.Dropout(0.5)

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
