import os
import argparse
import socket
import time
import json
import tqdm

from datetime import datetime
import numpy as np
import torch as th
import torch.nn.functional as F

import dgl
from dgl.data import register_data_args
from dgl.dataloading import DataLoader

from models.dist_sage import DistSAGE
from models.gat import GAT
from models.agnn import AGNN
from models.VA import VA
from models.gcn import GCN
from scripts.load_graph import OurDataset

os.environ["DGLBACKEND"] = "pytorch"


def compute_acc(pred, labels):
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


@th.no_grad()
def evaluate(model, g, val_nid, test_nid, batch_size, device):
    model.eval()

    dataloader_time = 0
    transfer_time = 0
    infer_time = 0
    transfer_back_time = 0
    get_feat_time = 0

    total_time = time.time()

    x = g.ndata["feat"]
    y_def = th.empty((g.number_of_nodes(), model.n_classes))

    for i, layer in enumerate(model.layers):
        if model.__class__.__name__ == "GAT" and i != len(model.layers) - 1:
            y = th.empty(
                (g.number_of_nodes(), model.n_hidden * model.heads[i])
            )
        else:
            y = y_def

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            graph=g,
            indices=th.arange(g.number_of_nodes(), dtype=g.idtype),
            graph_sampler=sampler,
            batch_size=args.batch_size_eval,
            shuffle=False,
            drop_last=False,
            pin_prefetcher=False,
        )

        t = time.time()

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            dataloader_time += time.time() - t

            t = time.time()
            block = blocks[0].to(device)
            transfer_time += time.time() - t

            t = time.time()
            h = x[input_nodes].to(device)
            h_dst = h[: block.number_of_dst_nodes()]
            get_feat_time += time.time() - t

            t = time.time()
            h = layer(block, (h, h_dst))

            if model.__class__.__name__ == "GAT":
                if i == len(model.layers) - 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)

            h = F.relu(h)
            infer_time += time.time() - t

            t = time.time()
            y[output_nodes] = h.cpu()
            transfer_back_time += time.time() - t

            t = time.time()
        x = y

    total_time = time.time() - total_time

    print(f"Dataloader time: {dataloader_time:.4f}")
    print(f"Transfer time: {transfer_time:.4f}")
    print(f"Inference time: {infer_time:.4f}")
    print(f"Trans back time: {transfer_back_time:.4f}")
    print(f"Get feat time: {get_feat_time:.4f}")
    above_sum = (
        dataloader_time
        + transfer_time
        + infer_time
        + transfer_back_time
        + get_feat_time
    )
    print(f"Above sum time: {above_sum:.4f}")

    labels = g.ndata["label"]
    t = time.time()
    val_acc = compute_acc(y[val_nid], labels[val_nid])
    test_acc = compute_acc(y[test_nid], labels[test_nid])
    print()
    print(f"Unnecessary stuff time: {time.time() - t:.4f}")

    model.train()
    return val_acc, test_acc, total_time


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data

    if args.model.lower() == "sage":
        model = DistSAGE(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
            F.relu,
            args.dropout,
        )
    elif args.model.lower() == "gcn":
        model = GCN(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
        )
    elif args.model.lower() == "va":
        model = VA(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
        )
    elif args.model.lower() == "agnn":
        model = AGNN(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,  # num of conv layers
            dropout_rate=0.5,
        )
    elif args.model.lower() == "gat":
        model = GAT(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
        )
    else:
        raise ValueError(
            f"We do not recognize the provided model name: {args.model}"
        )

    model.to(device, non_blocking=False)
    g.create_formats_()
    NUM_REPS = 3

    datetimes, eval_times = [], []
    forward_times = []
    backward_times = []
    sample_times = []

    if args.infer:
        for i in range(NUM_REPS):
            val_acc, test_acc, eval_time = evaluate(
                model,
                g,
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )

            print(
                "Rep {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format(
                    i, val_acc, test_acc, eval_time
                )
            )
            eval_times.append(eval_time)
            datetimes.append(str(datetime.now()))
    else:
        train_dataloader = DataLoader(
            g,
            indices=th.arange(g.number_of_nodes(), dtype=g.idtype),
            graph_sampler=dgl.dataloading.NeighborSampler([10, 25, 25]),
            device=device,
            batch_size=args.batch_size_eval,
            drop_last=False,
            num_workers=0,
            use_uva=True,
        )
        opt = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

        for i in range(args.num_epochs):
            forward_time = 0
            backward_time = 0
            sample_time = 0

            model.train()
            total_loss = 0
            sample_start = time.time()
            for it, (_, _, blocks) in enumerate(train_dataloader):
                sample_time = time.time() - sample_start
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]

                forward_start = time.time()
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                forward_end = time.time()
                opt.zero_grad()
                loss.backward()
                forward_time += forward_end - forward_start
                backward_time += time.time() - forward_end
                opt.step()
                total_loss += loss.item()

            val_acc, _, _ = evaluate(
                model, g, val_nid, test_nid, args.batch_size_eval, device)
            print(
                "Epoch {:05d} | Loss {:.4f} | Val accuracy {:.4f} ".format(
                    i, total_loss / (it + 1), val_acc
                )
            )

            eval_times.append(forward_time + backward_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            sample_times.append(sample_time)
            datetimes.append(str(datetime.now()))
            sample_start = time.time()

    res_file = f"{args.data_dir}/{args.graph_name}/node_norm_{args.model}.json"
    with open(res_file, "w") as f:
        json.dump(
            {
                "batch_size": args.batch_size_eval,
                "datetimes": datetimes,
                "eval_times": eval_times,
                "forward_times": forward_times,
                "backward_times": backward_times,
                "sample_times": sample_times,
            },
            f,
            indent=4,
        )

        return None


def main(args):
    print(socket.gethostname(), "Initializing DGL")

    start = time.time()
    graph_path = os.path.join(args.data_dir, args.graph_name)
    dataset = OurDataset(args.graph_name, raw_dir=graph_path)
    g = dataset[0]

    # from dgl.data import RedditDataset
    # data = RedditDataset(self_loop=True, raw_dir="data")
    # g = data[0]

    print(
        "load {} takes {:.3f} seconds".format(
            args.graph_name, time.time() - start
        )
    )
    print("|V|={}, |E|={}".format(g.number_of_nodes(), g.number_of_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )

    train_nid = g.ndata["train_mask"]
    val_nid = g.ndata["val_mask"]
    test_nid = g.ndata["test_mask"]

    device = th.device("cuda:0")
    n_classes = args.n_classes
    if n_classes == -1:
        labels = g.ndata["label"][np.arange(g.number_of_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["feat"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    register_data_args(parser)
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument(
        "-d", "--data_dir", type=str, default="data", help="data dir"
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=-1,
        help="The number of classes. If not specified, this"
        " value will be calculated via scaning all the labels"
        " in the dataset which probably causes memory burst.",
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size_eval", type=int, default=2**16)
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Do only inference on an untrained model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        help="model type: 'sage', 'va', 'gcn', 'agnn' or 'gat'",
    )
    args = parser.parse_args()

    # print(args)
    main(args)
