#!/bin/bash
PROGRAM=${1}

function create_script {
script_body="#!/bin/bash -l
#SBATCH --job-name="gnn"
#SBATCH --output="${MODEL}_${NODES}_${KRON_V}_${KRON_E}_${FEATS}.o"
#SBATCH --error="${MODEL}_${NODES}_${KRON_V}_${KRON_E}_${FEATS}.e"
#SBATCH --account="g34"
#SBATCH --time=00:40:00
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

module load daint-gpu
module load cudatoolkit
module load numpy

srun python ${PROGRAM}  -m ${MODEL} --layers 3 --warmup 1 --repeat 3 -d kronecker -v ${KRON_V} -e ${KRON_E} --features ${FEATS} --inference

srun python ${PROGRAM}  -m ${MODEL} --layers 3 --warmup 1 --repeat 3 -d kronecker -v ${KRON_V} -e ${KRON_E} --features ${FEATS}"

script_name="${MODEL}_${NODES}_${KRON_V}_${KRON_E}_${FEATS}.sbatch"
echo "${script_body}" > "${script_name}"
sbatch ${script_name}
}

TUPLES=(
    "131072 171798692"
    "262144 687194767"
    "1048576 109951163"
    "2097152 439804651"
)

NODESLIST=(1 4 16 64 256)
FEATs=(16 128)
MODELS=("VA" "GAT" "AGNN")

for feat in "${FEATs[@]}";do
    # Loop over the list of models
    for model in "${MODELS[@]}"; do
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
                export MODEL=$model
                create_script
            done
        done
    done
done    