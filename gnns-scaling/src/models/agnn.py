import torch
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

from dgl.utils import expand_as_pair
from dgl.base import DGLError
import dgl.function as fn

import dgl
import tqdm
from dgl.distributed import DistDataLoader

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from NeighborSampler import NeighborSampler


class AGNNConv(nn.Module):
    def __init__(self, init_beta=1.0, learn_beta=True, allow_zero_in_degree=True):
        super(AGNNConv, self).__init__()
        self._allow_zero_in_degree = allow_zero_in_degree
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor([init_beta]))
        else:
            self.register_buffer("beta", th.Tensor([init_beta]))

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute AGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *)` and :math:`(N_{out}, *)`, the :math:`*` in the later
            tensor must equal the previous one.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata["h"] = feat_src
            graph.srcdata["norm_h"] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple) or graph.is_block:
                graph.dstdata["norm_h"] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v("norm_h", "norm_h", "cos"))
            cos = graph.edata.pop("cos")
            e = self.beta * cos
            graph.edata["p"] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e("h", "p", "m"), fn.sum("m", "h"))
            return graph.dstdata.pop("h")


class AGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout_rate):
        super(AGNN, self).__init__()

        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(AGNNConv(learn_beta=True))

    def forward(self, blocks, x):
        h = x
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
            h = F.relu(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the AGNN model on full neighbors (i.e. without neighbor
        sampling).
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

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
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
