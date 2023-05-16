import argparse
from http import server
import time
import sys
import os
from xml.dom.NodeFilter import NodeFilter
import torch as th
import numpy as np
import dgl
import json
from load_graph import load_ogb, load_reddit, load_ours
import pyarrow
import pandas as pd
from pyarrow import csv

def parmetis_preproc(g: dgl.DGLGraph, dataset_name: str, num_parts: int, out_dir: str, num_trainers_per_machine):
    default_node_type = '_N'
    default_edge_type = '_E'
    print(f"Executing Preprocessing Step")

    '''
    store metadata of nodes
    <node_type_id> <node_weight_list> <type_wise_node_id>
    '''
    num_nodes = g.number_of_nodes()
    node_data = [np.zeros(num_nodes), np.ones(num_nodes), np.arange(num_nodes)]
    node_data = np.stack(node_data, axis=1)
    assert node_data.shape[1] == 3
    np.savetxt(os.path.join(out_dir, dataset_name + '_nodes.txt'), node_data, fmt='%d', delimiter=' ')

    '''
    Store the metadata of edges.
    <src_node_id> <dst_node_id> <type_wise_edge_id> <edge_type_id>
    ParMETIS cannot handle duplicated edges and self-loops. We should remove them
    in the preprocessing.
    '''
    src_id, dst_id = g.edges()
    # Remove self-loops
    self_loop_idx = src_id == dst_id
    not_self_loop_idx = src_id != dst_id
    self_loop_src_id = src_id[self_loop_idx]
    self_loop_dst_id = dst_id[self_loop_idx]
    self_loop_edge_id = th.arange(g.number_of_edges())[self_loop_idx]
    src_id = src_id[not_self_loop_idx]
    dst_id = dst_id[not_self_loop_idx]
    edge_id = th.arange(g.number_of_edges())[not_self_loop_idx]

    edge_data = th.stack([src_id, dst_id, edge_id, th.zeros_like(src_id)], 1)
    assert edge_data.shape[1] == 4
    np.savetxt(os.path.join(out_dir, dataset_name + "_edges.txt"), edge_data.numpy(), fmt='%d', delimiter=' ')

    removed_edge_data = th.stack([self_loop_src_id,
                                self_loop_dst_id,
                                self_loop_edge_id,
                                th.zeros_like(self_loop_src_id)],
                                1)
    np.savetxt(os.path.join(out_dir, dataset_name + "_removed_edges.txt"),
            removed_edge_data.numpy(), fmt='%d', delimiter=' ')
    print('There are {} edges, remove {} self-loops'.format(g.number_of_edges(), len(self_loop_src_id)))

    removed_edges = len(self_loop_src_id) > 0

    '''
    Note: Remove duplicated edges before input, too time consuming.

    # Remove duplicated edges.
    ids = (src_id * g.number_of_nodes() + dst_id).numpy()
    uniq_ids, idx = np.unique(ids, return_index=True)
    duplicate_idx = np.setdiff1d(np.arange(len(ids)), idx)
    duplicate_src_id = src_id[duplicate_idx]
    duplicate_dst_id = dst_id[duplicate_idx]
    duplicate_edge_id = edge_id[duplicate_idx]
    src_id = src_id[idx]
    dst_id = dst_id[idx]
    edge_id = edge_id[idx]

    removed_edge_data = th.stack([th.cat([self_loop_src_id, duplicate_src_id]),
                                th.cat([self_loop_dst_id, duplicate_dst_id]),
                                th.cat([self_loop_edge_id, duplicate_edge_id]),
                                th.cat([th.zeros_like(self_loop_src_id), th.zeros_like(duplicate_src_id)])],
                                1)
    np.savetxt(os.path.join(out_dir, dataset_name + "_removed_edges.txt"),
            removed_edge_data.numpy(), fmt='%d', delimiter=' ')
    print('There are {} edges, remove {} self-loops and {} duplicated edges'.format(g.number_of_edges(),
                                                                                    len(self_loop_src_id),
                                                                                    len(duplicate_src_id)))
    '''

    '''
    Store the basic metadata of the graph.
    <num_nodes> <num_edges> <total_node_weights>
    '''
    graph_stats = [g.number_of_nodes(), len(src_id), 1]
    with open(os.path.join(out_dir, dataset_name + "_stats.txt"), 'w') as filehandle:
        filehandle.writelines("{} {} {}".format(
            graph_stats[0], graph_stats[1], graph_stats[2]))


    # Store the ID ranges of nodes and edges of the entire graph.
    nid_ranges = {}
    eid_ranges = {}
    nid_ranges[default_node_type] = [0, int(g.number_of_nodes() + 1)]
    eid_ranges[default_edge_type] = [0, int(g.number_of_edges() + 1)]
    with open(os.path.join(out_dir, dataset_name + ".json"), 'w') as outfile:
        json.dump({'nid': nid_ranges, 'eid': eid_ranges}, outfile, indent=4)

    print(f"Done Preprocessing Step")

    return removed_edges

def parmetis_call(dataset_name: str, num_part: int, out_dir: str, num_part_per_proc: int):
    # Trigger ParMETIS for creating metis partitions for the input graph.
    parmetis_install_path = "pm_dglpart"
    parmetis_nfiles = os.path.join(out_dir, "parmetis_nfiles.txt")
    parmetis_efiles = os.path.join(out_dir, "parmetis_efiles.txt")
    parmetis_cmd = (
        f"srun -n {num_part} "
        f"{parmetis_install_path} {dataset_name} {num_part_per_proc}"
    )
    print(f"Executing ParMETIS: {parmetis_cmd}")
    cwd = os.getcwd()
    os.chdir(out_dir)
    os.system(parmetis_cmd)
    os.chdir(cwd)
    print(f"Done ParMETIS execution step")

def parmetis_postproc(orig_g: dgl.DGLGraph, in_dir: str, dataset_name: str, num_parts: int, removed_edges: bool):
    workspace_dir = in_dir
    out_dir = in_dir
    
    self_loop_edges = None
    if not removed_edges:
        removed_file = os.path.join(in_dir, dataset_name + "_removed_edges.txt")
        removed_df = csv.read_csv(removed_file, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                                parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        assert removed_df.num_columns == 4
        src_id = removed_df['f0'].to_numpy()
        edge_id = removed_df['f2'].to_numpy()
        self_loop_edges = [src_id, src_id, edge_id, np.zeros_like(src_id)]
        print('There are {} self-loops in the removed edges'.format(len(self_loop_edges[0])))
    
    schema_name = os.path.join(in_dir, dataset_name + ".json")
    with open(schema_name) as json_file:
        schema = json.load(json_file)
        
    nid_ranges = schema['nid']['_N']
    eid_ranges = schema['eid']['_E']

    def read_feats(file_name):
        attrs = csv.read_csv(file_name, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                            parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        num_cols = len(attrs.columns)
        return np.stack([attrs.columns[i].to_numpy() for i in range(num_cols)], 1)

    max_nid = np.iinfo(np.int32).max
    num_edges = 0
    num_nodes = 0
    node_map_val = []
    edge_map_val = []

    for part_id in range(num_parts):
        part_dir = out_dir + '/part' + str(part_id)
        os.makedirs(part_dir, exist_ok=True)
        node_file = 'p{:03}-{}_nodes.txt'.format(part_id, dataset_name)
        '''
        <node_id> <node_type_id> <node_weight_list> <type_wise_node_id>
        '''
        orig_type_nid_col = 4
        nodes = csv.read_csv(os.path.join(in_dir, node_file), read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                         parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        nids, global_nids = nodes.columns[0].to_numpy(), nodes.columns[3].to_numpy()
        assert np.all(nids[1:] - nids[:-1] == 1)
        
        nid_range = (nids[0], nids[-1])
        global_nid_range = (global_nids[0], global_nids[-1])
        num_nodes += len(nodes)

        node_map_val.append([int(nids[0]), int(nids[-1] + 1)])

        edge_file = 'p{:03}-{}_edges.txt'.format(part_id, dataset_name)

        '''
        <src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_edge_id> <edge_type> <attributes>
        '''

        edges = csv.read_csv(os.path.join(in_dir, edge_file), read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                            parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        num_cols = len(edges.columns)
        src_id, dst_id, orig_src_id, orig_dst_id, orig_edge_id, etype_ids = [
            edges.columns[i].to_numpy() for i in range(num_cols)]
        
        src_id_list, dst_id_list = [src_id], [dst_id]
        orig_src_id_list, orig_dst_id_list = [orig_src_id], [orig_dst_id]
        orig_edge_id_list, etype_id_list = [orig_edge_id], [etype_ids]

        if self_loop_edges is not None and len(self_loop_edges[0]) > 0:
            uniq_orig_nids, idx = np.unique(orig_dst_id, return_index=True)
            common_nids, common_idx1, common_idx2 = np.intersect1d(
                uniq_orig_nids, self_loop_edges[0], return_indices=True)
            idx = idx[common_idx1]
            # the IDs after ID assignment
            src_id_list.append(dst_id[idx])
            dst_id_list.append(dst_id[idx])
            # homogeneous IDs in the input graph.
            orig_src_id_list.append(self_loop_edges[0][common_idx2])
            orig_dst_id_list.append(self_loop_edges[0][common_idx2])
            # edge IDs and edge type.
            orig_edge_id_list.append(self_loop_edges[2][common_idx2])
            etype_id_list.append(self_loop_edges[3][common_idx2])
            print('Add {} self-loops in partition {}'.format(len(idx), part_id))
        
        src_id = np.concatenate(src_id_list) if len(
            src_id_list) > 1 else src_id_list[0]
        dst_id = np.concatenate(dst_id_list) if len(
            dst_id_list) > 1 else dst_id_list[0]
        orig_src_id = np.concatenate(orig_src_id_list) if len(
            orig_src_id_list) > 1 else orig_src_id_list[0]
        orig_dst_id = np.concatenate(orig_dst_id_list) if len(
            orig_dst_id_list) > 1 else orig_dst_id_list[0]
        orig_edge_id = np.concatenate(orig_edge_id_list) if len(
            orig_edge_id_list) > 1 else orig_edge_id_list[0]
        etype_ids = np.concatenate(etype_id_list) if len(
            etype_id_list) > 1 else etype_id_list[0]
        print('There are {} edges in partition {}'.format(len(src_id), part_id))

        edge_map_val.append([int(num_edges), int(num_edges + len(etype_ids))])

        
        ids = np.concatenate(
            [src_id, dst_id, np.arange(nid_range[0], nid_range[1] + 1)])
        uniq_ids, idx, inverse_idx = np.unique(
            ids, return_index=True, return_inverse=True)
        assert len(uniq_ids) == len(idx)
        # We get the edge list with their node IDs mapped to a contiguous ID range.
        local_src_id, local_dst_id = np.split(inverse_idx[:len(src_id) * 2], 2)
        compact_g = dgl.graph((local_src_id, local_dst_id))
        compact_g.edata['orig_id'] = th.as_tensor(orig_edge_id)
        compact_g.edata[dgl.ETYPE] = th.zeros(compact_g.number_of_edges())
        compact_g.edata['inner_edge'] = th.ones(compact_g.number_of_edges(), dtype=th.bool)

            
        orig_ids = np.concatenate([orig_src_id, orig_dst_id, np.arange(global_nid_range[0], global_nid_range[1] + 1)])
        orig_homo_ids = orig_ids[idx]
        compact_g.ndata['orig_id'] = th.as_tensor(orig_homo_ids)
        compact_g.ndata[dgl.NTYPE] = th.zeros(compact_g.number_of_nodes())
        compact_g.ndata[dgl.NID] = th.as_tensor(uniq_ids)
        compact_g.ndata['inner_node'] = th.as_tensor(np.logical_and(
            uniq_ids >= nid_range[0], uniq_ids <= nid_range[1]))
        local_nids = compact_g.ndata[dgl.NID][compact_g.ndata['inner_node'].bool()]
        assert np.all((local_nids == th.arange(
            local_nids[0], local_nids[-1] + 1)).numpy())
        print('|V|={}'.format(compact_g.number_of_nodes()))
        print('|E|={}'.format(compact_g.number_of_edges()))

        # We need to reshuffle nodes in a partition so that all local nodes are labelled starting from 0.
        reshuffle_nodes = th.arange(compact_g.number_of_nodes())
        reshuffle_nodes = th.cat([reshuffle_nodes[compact_g.ndata['inner_node'].bool()],
                                reshuffle_nodes[compact_g.ndata['inner_node'] == 0]])
        compact_g1 = dgl.node_subgraph(compact_g, reshuffle_nodes)
        compact_g1.ndata['orig_id'] = compact_g.ndata['orig_id'][reshuffle_nodes]
        compact_g1.ndata[dgl.NTYPE] = compact_g.ndata[dgl.NTYPE][reshuffle_nodes]
        compact_g1.ndata[dgl.NID] = compact_g.ndata[dgl.NID][reshuffle_nodes]
        compact_g1.ndata['inner_node'] = compact_g.ndata['inner_node'][reshuffle_nodes]
        compact_g1.edata['orig_id'] = compact_g.edata['orig_id'][compact_g1.edata[dgl.EID]]
        compact_g1.edata[dgl.ETYPE] = compact_g.edata[dgl.ETYPE][compact_g1.edata[dgl.EID]]
        compact_g1.edata['inner_edge'] = compact_g.edata['inner_edge'][compact_g1.edata[dgl.EID]]

        
        # reshuffle edges on ETYPE as node_subgraph relabels edges
        idx = th.argsort(compact_g1.edata[dgl.ETYPE])
        u, v = compact_g1.edges()
        u = u[idx]
        v = v[idx]
        compact_g2 = dgl.graph((u, v))
        compact_g2.ndata['orig_id'] = compact_g1.ndata['orig_id']
        compact_g2.ndata[dgl.NTYPE] = compact_g1.ndata[dgl.NTYPE]
        compact_g2.ndata[dgl.NID] = compact_g1.ndata[dgl.NID]
        compact_g2.ndata['inner_node'] = compact_g1.ndata['inner_node']
        compact_g2.edata['orig_id'] = compact_g1.edata['orig_id'][idx]
        compact_g2.edata[dgl.ETYPE] = compact_g1.edata[dgl.ETYPE][idx]
        compact_g2.edata['inner_edge'] = compact_g1.edata['inner_edge'][idx]
        compact_g2.edata[dgl.EID] = th.arange(
            num_edges, num_edges + compact_g2.number_of_edges())
        num_edges += compact_g2.number_of_edges()

        dgl.save_graphs(part_dir + '/graph.dgl', [compact_g2])

        node_data = {}
        local_node_idx = compact_g2.ndata['inner_node'].bool()
        local_nodes = compact_g2.ndata['orig_id'][local_node_idx].numpy()
        for name in orig_g.nodes['_N'].data:
            node_data['_N' + '/' + name] = orig_g.nodes['_N'].data[name][local_nodes]
        print('node features:', node_data.keys())
        dgl.data.utils.save_tensors(part_dir + '/node_feat.dgl', node_data)

        edge_data = {}
        local_edges = compact_g2.edata['orig_id']
        for name in orig_g.edges['_E'].data:
            edge_data['_E' + '/' + name] = orig_g.edges['_E'].data[name][local_edges]
        print('edge features:', edge_data.keys())
        dgl.data.utils.save_tensors(part_dir + '/edge_feat.dgl', edge_data)
    
    part_metadata = {'graph_name': dataset_name,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'part_method': 'metis',
                    'num_parts': num_parts,
                    'halo_hops': 1,
                    'node_map': {'_N': node_map_val},
                    'edge_map': {'_E': edge_map_val},
                    'ntypes': {'_N': 0},
                    'etypes': {'_E': 0}}

    for part_id in range(num_parts):
        part_dir = 'part' + str(part_id)
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                    'edge_feats': edge_feat_file,
                                                    'part_graph': part_graph_file}
    with open('{}/{}.json'.format(out_dir, dataset_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

def parmetis_run(g: dgl.DGLGraph, dataset_name: str, num_parts: int, out_dir: str, num_trainers_per_machine, num_part_per_proc: int):
    '''
    Only support homogeneous graph right now.

    '''
    removed_edges = parmetis_preproc(g, dataset_name, num_parts, out_dir, num_trainers_per_machine)
    parmetis_call(dataset_name, num_parts, out_dir, num_part_per_proc)
    parmetis_postproc(g, out_dir, dataset_name, num_parts, removed_edges)
    os.system('rm ' + out_dir + '/*.txt')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogb-product, ogb-paper100M",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--num_part_per_proc",
        type=int,
        default=1,
        help="the number of graph parts per process generate",
    )
    argparser.add_argument(
        "--no_dup_edge",
        action="store_true",
        help="Make sure no duplicate edges in input",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    if args.no_dup_edge:
        pass
    else:
        sys.exit("Please verify there is no duplicate edge in graph, and add flag --no_dup_edge")

    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_reddit(root=args.output)
    elif args.dataset == "ogb-product":
        g, _ = load_ogb("ogbn-products", root=args.output)
    elif args.dataset == "ogb-paper100M":
        g, _ = load_ogb("ogbn-papers100M", root=args.output)
    else:
        g, _ = load_ours(args.dataset, root=args.output)
    print(
        "load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.number_of_nodes(), g.number_of_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    parmetis_run(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        num_trainers_per_machine=args.num_trainers_per_machine,
        num_part_per_proc=args.num_part_per_proc
    )
