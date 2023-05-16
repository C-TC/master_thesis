#!/bin/bash

function create_script {
script_body="#!/bin/bash -l
#SBATCH --job-name="gnn"
#SBATCH --output="${MODEL}_CSCMM_${KRON_V}_${KRON_E}.o"
#SBATCH --error="${MODEL}_CSCMM_${KRON_V}_${KRON_E}.e"
#SBATCH --account="g34"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

module load daint-gpu
module load cudatoolkit
module load numpy


srun python unified_single_bench.py  -m ${MODEL} -d kronecker -v ${KRON_V} -e ${KRON_E} --features 16
srun python unified_single_bench.py  -m ${MODEL} -d kronecker -v ${KRON_V} -e ${KRON_E} --features 128"

script_name="${MODEL}_CSCMM_${KRON_V}_${KRON_E}.sbatch"
echo "${script_body}" > "${script_name}"
sbatch ${script_name}
}

TUPLES=(
    "131072 1717987"
    "131072 17179869"
    "262144 6871948"
    "262144 68719477"
    "524288 27487791"
    "524288 274877907"
)

MODELS=("VA" "GAT" "AGNN")

# Loop over the list of models
for model in "${MODELS[@]}"; do
    # Loop over the list of lists
    for STRING in "${TUPLES[@]}"; do
            # Extract the values for KRON_V, KRON_E
            KRON_V=$(echo $STRING | cut -d " " -f 1)
            KRON_E=$(echo $STRING | cut -d " " -f 2)

            # Export the variables and launch the sbatch file
            export KRON_V=$KRON_V
            export KRON_E=$KRON_E
            export FEATS=$feat
            export MODEL=$model
            create_script
    done
done