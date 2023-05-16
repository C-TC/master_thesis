#!/bin/bash -l
#SBATCH --job-name="kron-graph-partitioning"
#SBATCH --account="g34"
#SBATCH --nodes=1
#SBATCH --mem=1T
#SBATCH --time=6:00:00
#SBATCH --partition=fat
#SBATCH --output=kron_part.txt
#SBATCH --error=kron_part.txt

# ault does not have scratch
PROJECT_ROOT=$HOME/gnns-scaling
cd $PROJECT_ROOT

conda activate $PROJECT_ROOT/venv

DATA_DIR=data/kron

GRAPHS=(
    s-17_e-1311_s-0.01
    s-18_e-2622_s-0.01
    s-20_e-105_s-0.0001
    s-21_e-210_s-0.0001
)

#NODES=(1 2 4 8 16)
NODES=(1)

for GRAPH in "${GRAPHS[@]}"; do
    IFS='_-' read -r -a PARAMS <<< "$GRAPH"

    for NODE in "${NODES[@]}"; do

        DATASET="n${NODE}_a${PARAMS[1]}_e${PARAMS[3]}_s${PARAMS[5]}"
        printf "${DATASET} " && [[ -d "${DATA_DIR}"/"${DATASET}" ]] && echo "OK" || echo "NOPE"

        OUTPATH="$DATA_DIR"
        cmd1=$(
            printf "srun python $PROJECT_ROOT/src/scripts/partition_graph.py 
                --balance_edges
                --dataset $DATASET
                --output $DATA_DIR/$DATASET
                --num_parts $NODE
        ")

        [[ ! -z $cmd ]] && cmd+=" && \n"
        cmd+="$cmd1"
    done
done
cmd="$(echo -e $cmd)"
echo "$cmd"
eval "$cmd"
