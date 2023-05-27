import numpy as np
from scipy import sparse
import dace
from typing import List, Tuple
from dace.transformation.dataflow import MapInterchange, StripMining, MapReduceFusion, MapExpansion, MapToForLoop, TrivialTaskletElimination, GPUGridStridedTiling, TaskletFusion


M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
nnz = dace.symbol('nnz')


@dace.program
def VA_f_0(out_data: dace.float32[nnz], indices: dace.int32[nnz],
           indptr: dace.int32[M + 1], H_tile_1: dace.float32[M, K],
           H_tile_2: dace.float32[N, K]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            colNo = indices[j]
            for k in dace.map[0:K]:
                out_data[j] += H_tile_1[rowNo, k] * H_tile_2[colNo, k]


# GAT


@dace.program
def GAT_f_0(out_data: dace.float32[nnz], out_row_max: dace.float32[M],
            indices: dace.int32[nnz], indptr: dace.int32[M + 1],
            l: dace.float32[M], r: dace.float32[N], num_rows: dace.int32):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            colNo = indices[j]
            tmp = l[rowNo] + r[colNo]
            tmp = np.maximum(tmp, 0.2 * tmp)
            out_data[j] = tmp
            out_row_max[rowNo] = np.maximum(out_row_max[rowNo], tmp)


@dace.program
def GAT_f_1(out_data: dace.float32[nnz], row_sum: dace.float32[M],
            E_data: dace.float32[nnz], indices: dace.int32[nnz],
            indptr: dace.int32[M + 1], row_max: dace.float32[M]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            tmp = np.exp(E_data[j] - row_max[rowNo])
            out_data[j] = tmp
            row_sum[rowNo] += tmp


@dace.program
def GAT_f_2(alpha_data: dace.float32[nnz], row_sum: dace.float32[M],
            indices: dace.int32[nnz], indptr: dace.int32[M + 1]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            alpha_data[j] /= row_sum[rowNo]


@dace.program
def GAT_b_0(out_data: dace.float32[nnz], row_dot: dace.float32[M],
            Alpha_data: dace.float32[nnz], dZ: dace.float32[M, K],
            M_mat: dace.float32[M, K], indices: dace.int32[nnz],
            indptr: dace.int32[M + 1]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            colNo = indices[j]
            for k in dace.map[0:K]:
                out_data[j] += dZ[rowNo, k] * M_mat[colNo, k]
            row_dot[rowNo] += out_data[j] * Alpha_data[j]


@dace.program
def GAT_b_1(dl: dace.float32[M], dr: dace.float32[N], l: dace.float32[M],
            r: dace.float32[N], Alpha_data: dace.float32[nnz],
            dAlpha_data: dace.float32[nnz], indices: dace.int32[nnz],
            indptr: dace.int32[M + 1], row_dot: dace.float32[M]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            colNo = indices[j]
            dE = (dAlpha_data[j] - row_dot[rowNo]) * Alpha_data[j]
            D_recomp = l[rowNo] + r[colNo]
            dE = np.where(D_recomp <= 0, dE * 0.2, dE)
            dl[rowNo] += dE
            dr[colNo] += dE


@dace.program
def AGNN_f_0(out_data: dace.float32[nnz], indices: dace.int32[nnz],
             indptr: dace.int32[M + 1], H_tile_1: dace.float32[M, K],
             H_tile_2: dace.float32[N, K], H_tile_1_norm: dace.float32[M],
             H_tile_2_norm: dace.float32[N]):
    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            colNo = indices[j]
            tmp = 0.0
            for k in dace.map[0:K]:
                tmp += (H_tile_1[rowNo, k] * H_tile_2[colNo, k])
            out_data[j] = tmp / (H_tile_1_norm[rowNo] * H_tile_2_norm[colNo])


@dace.program
def AGNN_b_0(dC_out_data: dace.float32[nnz], dD_out_data: dace.float32[nnz],
             indices: dace.int32[nnz], indptr: dace.int32[M + 1],
             dZ: dace.float32[M, K], M_mat: dace.float32[N, K],
             H_tile_1: dace.float32[M, K], H_tile_2: dace.float32[N, K],
             n_tile_1: dace.float32[M], n_tile_2: dace.float32[N]):

    for i in dace.map[0:M]:
        for j in dace.map[indptr[i]:indptr[i + 1]]:
            rowNo = i
            colNo = indices[j]
            dQ = 0.0
            C = 0.0
            D = n_tile_1[rowNo] * n_tile_2[colNo]
            for k in dace.map[0:K]:
                dQ += dZ[rowNo, k] * M_mat[colNo, k]
            for k in dace.map[0:K]:
                C += H_tile_1[rowNo, k] * H_tile_2[colNo, k]

            dC_out_data[j] = dQ / D
            dD_out_data[j] = -C * dQ / (D * D)





def copy_to_gpu(sdfg):
    for k, v in sdfg.arrays.items():
        if not v.transient and isinstance(v, dace.data.Array):
            v.storage = dace.dtypes.StorageType.GPU_Global


def find_map_entry(sdfg: dace.SDFG, map_name_list: List[str]) -> Tuple[dace.sdfg.nodes.MapEntry]:
    if isinstance(map_name_list, str):
        map_name_list = [
            map_name_list,
        ]
    ret_list = [None] * len(map_name_list)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                for i, map_name in enumerate(map_name_list):
                    if map_name == node.map.params[0]:
                        ret_list[i] = node
    # check if all map entries are found
    assert all([x is not None for x in ret_list])

    # unpack if only one map entry is found
    if len(ret_list) == 1:
        return ret_list[0]
    else:
        return tuple(ret_list)


def GPU_normal_no_atomic(sdfg: dace.SDFG, atomic_arrays: List[str] = [], missing_wcr_arrays={}) -> None:
    
    sdfg.simplify()

    ime, jme = find_map_entry(sdfg, ["i", "j"])
    sdfg.apply_transformations_repeated(TaskletFusion)

    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    copy_to_gpu(sdfg)
    ime.map.schedule = dace.ScheduleType.GPU_Device
    jme.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic

    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.data in missing_wcr_arrays:
            e.data.wcr = missing_wcr_arrays[e.data.data]

    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.wcr:
            # print(e.data.data)
            if e.data.data not in atomic_arrays:
                e.data.wcr_nonatomic = True

    sdfg.validate()

def GPU_strided_no_atomic(sdfg: dace.SDFG, atomic_arrays: List[str] = [], missing_wcr_arrays={}) -> None:
    
    sdfg.simplify()

    ime, jme = find_map_entry(sdfg, ["i", "j"])
    sdfg.apply_transformations_repeated(TaskletFusion)

    sdfg.apply_transformations_repeated(TrivialTaskletElimination)

    copy_to_gpu(sdfg)
    GPUGridStridedTiling.apply_to(sdfg, outer_map_entry=ime, inner_map_entry=jme)

    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.data in missing_wcr_arrays:
            e.data.wcr = missing_wcr_arrays[e.data.data]

    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and e.data.wcr:
            # print(e.data.data)
            if e.data.data not in atomic_arrays:
                e.data.wcr_nonatomic = True

    sdfg.validate()


DACE_GPU_NORMAL = {
    "VA_f_0": GPU_normal_no_atomic,
    "GAT_f_0": (lambda a: GPU_normal_no_atomic(a, ['__tmp_40_12_w'], {'__tmp_40_12_w': 'lambda a, b: max(a, b)'})),
    "GAT_f_1": (lambda a: GPU_normal_no_atomic(a, ['row_sum'])),
    "GAT_f_2": GPU_normal_no_atomic,
    "GAT_b_0": (lambda a: GPU_normal_no_atomic(a, ['__tmp_75_12_w'])),
    "GAT_b_1": (lambda a: GPU_normal_no_atomic(a, ['__tmp_90_12_w', '__tmp_91_12_w'])), 
    "AGNN_f_0": GPU_normal_no_atomic,
    "AGNN_b_0": GPU_normal_no_atomic,
}


DACE_GPU_STRIDED = {
    "VA_f_0": GPU_strided_no_atomic,
    "GAT_f_0": (lambda a: GPU_strided_no_atomic(a, ['__tmp_40_12_w'], {'__tmp_40_12_w': 'lambda a, b: max(a, b)'})),
    "GAT_f_1": (lambda a: GPU_strided_no_atomic(a, ['row_sum'])),
    "GAT_f_2": GPU_strided_no_atomic,
    "GAT_b_0": (lambda a: GPU_strided_no_atomic(a, ['__tmp_75_12_w'])),
    "GAT_b_1": (lambda a: GPU_strided_no_atomic(a, ['__tmp_90_12_w', '__tmp_91_12_w'])), 
    "AGNN_f_0": GPU_strided_no_atomic,
    "AGNN_b_0": GPU_strided_no_atomic,
}
