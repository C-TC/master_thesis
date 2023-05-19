import dace
from dace.transformation.dataflow import LiftEinsum, MapFission, TaskletFusion
from dace.transformation.optimizer import Optimizer
from dace.transformation.auto.auto_optimize import auto_optimize


B1_dimension = dace.symbol('B1_dimension')
B2_dimension = dace.symbol('B2_dimension')
size_A2_crd = dace.symbol('size_A2_crd')
size_A2_pos = dace.symbol('size_A2_pos')
size_A_vals = dace.symbol('size_A_vals')
size_B_vals = dace.symbol('size_B_vals')
size_D_vals = dace.symbol('size_D_vals')


# The program has the following changes
# 1. D_vals is initialized once before the loops/Maps
# 2. Explicit WCR (augmented assignment) is used
# 3. A2_crd and A2_pos are now int64 to allow ScalarToSymbolPromotion


@dace.program
def gnn_sddmm1compute(A2_crd: dace.int64[size_A2_crd], A2_pos: dace.int64[size_A2_pos], A_vals: dace.float64[size_A_vals], B_vals: dace.float64[size_B_vals], D_vals: dace.float64[size_D_vals]):

    D_vals[:] = 0
    for i in dace.map[0: B1_dimension: 1]:
        for jA in dace.map[A2_pos[i]: A2_pos[(i + 1)]: 1]:
            j = A2_crd[jA]
            for k in dace.map[0: B1_dimension: 1]:
                kB = i * B2_dimension + k
                jB = k * B2_dimension + j
                D_vals[jA] += (A_vals[jA] * B_vals[kB]) * B_vals[jB]


sdfg = gnn_sddmm1compute.to_sdfg(simplify=True)

sdfg.view()
applied = True
while applied:
    applied = False
    for xform in Optimizer(sdfg).get_pattern_matches(patterns=[TaskletFusion]):
        if xform.t2.label.startswith('assign'):
            continue
        csdfg = sdfg.sdfg_list[xform.sdfg_id]
        cgraph = csdfg.nodes()[xform.state_id]
        xform.apply(cgraph, csdfg)
        applied = True
        break

sdfg.view()
sdfg.apply_transformations_repeated(MapFission)
sdfg.view()
sdfg.apply_transformations_repeated(LiftEinsum)
sdfg.view()