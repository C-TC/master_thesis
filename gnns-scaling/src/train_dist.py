import os
import argparse
import socket
import time
import json

from datetime import datetime
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
from dgl.data import register_data_args
from dgl.distributed import DistDataLoader

from NeighborSampler import NeighborSampler
from models.dist_sage import DistSAGE
from models.gat import GAT
from models.agnn import AGNN
from models.VA import VA
from models.gcn import GCN

os.environ["DGLBACKEND"] = "pytorch"


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        start_time = time.time()
        pred = model.inference(g, inputs, batch_size, device)
        infer_time = time.time() - start_time
    model.train()

    t = time.time()
    val_acc = compute_acc(pred[val_nid], labels[val_nid])
    test_acc = compute_acc(pred[test_nid], labels[test_nid])
    print("Unnecessary stuff time: {:.4f}".format(time.time() - t))

    return val_acc, test_acc, infer_time


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    shuffle = True
    # Create sampler
    sampler = NeighborSampler(
        g,
        [int(fanout) for fanout in args.fan_out.split(",")],
        dgl.distributed.sample_neighbors,
        device,
    )

    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=shuffle,
        drop_last=False,
    )

    # Define model and optimizer
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

    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device
            )

    datetimes, eval_times = [], []

    if args.infer:
        for i in range(3):
            start = time.time()
            val_acc, test_acc, infer_time = evaluate(
                model.module,
                g,
                g.ndata["features"],
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )
            print(
                "Rep {}, Part {}, time: {:.4f}".format(i, g.rank(), infer_time)
            )
            eval_times.append(infer_time)
            datetimes.append(str(datetime.now()))

    else:
        loss_fcn = nn.CrossEntropyLoss()
        loss_fcn = loss_fcn.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        iter_tput = []
        # Times
        forward_times = []
        backward_times = []
        sample_times = []
        epoch = 0
        for epoch in range(args.num_epochs):
            sample_time = 0
            forward_time = 0
            backward_time = 0
            update_time = 0
            num_seeds = 0
            num_inputs = 0
            start = time.time()
            # Loop over the dataloader to sample the computation dependency graph
            # as a list of blocks.
            step_time = []

            tic = time.time()

            with model.join():
                for step, blocks in enumerate(dataloader):
                    tic_step = time.time()
                    sample_time += tic_step - start

                    # The nodes for input lies at the LHS side of the first block.
                    # The nodes for output lies at the RHS side of the last block.
                    batch_inputs = blocks[0].srcdata["features"]
                    batch_labels = blocks[-1].dstdata["labels"]
                    batch_labels = batch_labels.long()

                    num_seeds += len(blocks[-1].dstdata[dgl.NID])
                    num_inputs += len(blocks[0].srcdata[dgl.NID])
                    blocks = [block.to(device) for block in blocks]
                    batch_labels = batch_labels.to(device)

                    # Compute loss and prediction
                    start = time.time()
                    batch_pred = model(blocks, batch_inputs)
                    loss = loss_fcn(batch_pred, batch_labels)
                    forward_end = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    compute_end = time.time()
                    forward_time += forward_end - start
                    backward_time += compute_end - forward_end

                    optimizer.step()
                    update_time += time.time() - compute_end

                    step_t = time.time() - tic_step
                    step_time.append(step_t)
                    iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                    if step % args.log_every == 0:
                        acc = compute_acc(batch_pred, batch_labels)
                        gpu_mem_alloc = (
                            th.cuda.max_memory_allocated() / 1000000
                            if th.cuda.is_available()
                            else 0
                        )
                        print(
                            "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB | time {:.3f} s".format(
                                g.rank(),
                                epoch,
                                step,
                                loss.item(),
                                acc.item(),
                                np.mean(iter_tput[3:]),
                                gpu_mem_alloc,
                                np.sum(step_time[-args.log_every :]),
                            )
                        )
                    start = time.time()

            toc = time.time()
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            sample_times.append(sample_time)
            eval_times.append(forward_time + backward_time)
            datetimes.append(str(datetime.now()))

            print(
                "Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}".format(
                    g.rank(),
                    toc - tic,
                    sample_time,
                    forward_time,
                    backward_time,
                    update_time,
                    num_seeds,
                    num_inputs,
                )
            )
            epoch += 1

            if epoch % args.eval_every == 0 and epoch != 0:
                start = time.time()
                val_acc, test_acc, _ = evaluate(
                    model.module,
                    g,
                    g.ndata["features"],
                    g.ndata["labels"],
                    val_nid,
                    test_nid,
                    args.batch_size_eval,
                    device,
                )
                print(
                    "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format(
                        g.rank(), val_acc, test_acc, time.time() - start
                    )
                )

    res_file = f"{args.data_dir}/{args.graph_name}/node{g.rank():02d}_{args.model}.json"
    with open(res_file, "w") as f:
        json.dump(
            {
                "rank": g.rank(),
                "batch_size": args.batch_size,
                "datetimes": datetimes,
                "eval_times": eval_times,  # sum of forward and backward
                "forward_times": forward_times,
                "backward_times": backward_times,
                "sample_times": sample_times,
            },
            f,
            indent=4,
        )

def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(
        args.graph_name, part_config=args.part_config
    )
    print(socket.gethostname(), "rank:", g.rank())

    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        idx = pb.partid2nids(pb.partid)
        g.ndata["train_mask"][idx] = th.ones(len(idx), dtype=th.uint8)
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    if n_classes == -1:
        labels = g.ndata["labels"][np.arange(g.number_of_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["features"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    register_data_args(parser)
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument(
        "-d", "--data_dir", type=str, default="data", help="data dir"
    )
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--num_clients", type=int, help="The number of clients"
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=-1,
        help="The number of classes. If not specified, this"
        " value will be calculated via scaning all the labels"
        " in the dataset which probably causes memory burst.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="10,25,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num of batches to be the same.",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="socket",
        help="backend net type, 'socket' or 'tensorpipe'",
    )
    parser.add_argument(
        "--infer",
        default=False,
        action="store_true",
        help="Do only inference on an untrained model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        help="model type: 'sage', 'gcn', 'va', 'agnn' or 'gat'",
    )
    args = parser.parse_args()

    # print(args)
    main(args)
