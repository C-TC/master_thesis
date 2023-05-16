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
    "1 131072 1717987"
    "1 131072 17179869"
    "1 131072 171798692"
    "4 262144 6871948"
    "4 262144 68719477"
    "4 262144 687194767"
    "16 524288 27487791"
    "16 524288 274877907"
    "16 524288 2748779069"
    "64 1048576 109951163"
    "64 1048576 1099511628"
    "64 1048576 10995116278"
    "256 2097152 439804651"
    "256 2097152 4398046511"
    "256 2097152 43980465111"
)

FEATs=(16 128)
MODELS=("VA" "GAT" "AGNN")

for feat in "${FEATs[@]}";do
    # Loop over the list of models
    for model in "${MODELS[@]}"; do
        # Loop over the list of lists
        for STRING in "${TUPLES[@]}"; do
        # Extract the values for KRON_V, KRON_E, and FEATS
                NODES=$(echo $STRING | cut -d " " -f 1)
                KRON_V=$(echo $STRING | cut -d " " -f 2)
                KRON_E=$(echo $STRING | cut -d " " -f 3)

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