#!/bin/bash
#SBATCH --job-name="cscmm"
#SBATCH --account="g34"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=ault24
#SBATCH --partition=intelv100

module load cuda
conda init bash
conda activate thesis

TUPLES=(
    "131072 1717987"
    "131072 17179869"
    "131072 171798692"
    "262144 6871948"
    "262144 68719477"
    "262144 687194767"
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
            
            python unified_single_bench.py  -m ${MODEL} -d kronecker -v ${KRON_V} -e ${KRON_E} --features 16
            python unified_single_bench.py  -m ${MODEL} -d kronecker -v ${KRON_V} -e ${KRON_E} --features 128
    done
done
