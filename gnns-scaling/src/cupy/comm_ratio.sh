#!/bin/bash

function create_script {
script_body="#!/bin/bash -l
#SBATCH --job-name="gnn"
#SBATCH --output="${NODES}_${KRON_V}_${KRON_E}_${FEATS}.o"
#SBATCH --error="${NODES}_${KRON_V}_${KRON_E}_${FEATS}.e"
#SBATCH --account="g34"
#SBATCH --time=01:00:00
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

module load daint-gpu
module load cudatoolkit
module load numpy

srun python bench_comm_ratio.py  -m VA -d kronecker -v ${KRON_V} -e ${KRON_E} --features 16
srun python bench_comm_ratio.py  -m GAT -d kronecker -v ${KRON_V} -e ${KRON_E} --features 16
srun python bench_comm_ratio.py  -m AGNN -d kronecker -v ${KRON_V} -e ${KRON_E} --features 16

srun python bench_comm_ratio.py  -m VA -d kronecker -v ${KRON_V} -e ${KRON_E} --features 128
srun python bench_comm_ratio.py  -m GAT -d kronecker -v ${KRON_V} -e ${KRON_E} --features 128
srun python bench_comm_ratio.py  -m AGNN -d kronecker -v ${KRON_V} -e ${KRON_E} --features 128"

script_name="${NODES}_${KRON_V}_${KRON_E}_${FEATS}.sbatch"
echo "${script_body}" > "${script_name}"
sbatch ${script_name}
}

TUPLES=(
    "262144 6871948"
    "262144 68719477"
    "262144 687194767"
)

NODESLIST=(4 16 64)

# Loop over the list of lists
for STRING in "${TUPLES[@]}"; do
    for NODES in "${NODESLIST[@]}"; do
        # Extract the values for KRON_V, KRON_E
        KRON_V=$(echo $STRING | cut -d " " -f 1)
        KRON_E=$(echo $STRING | cut -d " " -f 2)

        # Export the variables and launch the sbatch file
        export NODES=$NODES
        export KRON_V=$KRON_V
        export KRON_E=$KRON_E
        export FEATS=$feat
        create_script
    done
done