#!/bin/bash
#SBATCH --job-name="kernel_bench"
#SBATCH --account="g34"
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=ault24
#SBATCH --partition=intelv100

module load cuda/11.8.0
conda init bash
conda activate thesis

TUPLES=(
    "131072 1717987"
    "131072 17179869"
    "262144 6871948"
    "262144 68719477"
    "524288 27487791"
    "524288 274877907"
)

DATASETS=("random" "kronecker")

# Loop over the list of DATASETS
for dataset in "${DATASETS[@]}"; do
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
            
            python benchmark.py  -d ${dataset} -v ${KRON_V} -e ${KRON_E} --features 16
            python benchmark.py  -d ${dataset} -v ${KRON_V} -e ${KRON_E} --features 128
    done
done
