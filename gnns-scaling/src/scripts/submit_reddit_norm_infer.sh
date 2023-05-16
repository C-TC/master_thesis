#!/bin/bash -l
#SBATCH --job-name="kronecker_DGL"
#SBATCH --account="g34"
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=exp_norm_s.log
#SBATCH --error=exp_norm_s.log

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

module load daint-gpu

PROJECT_ROOT=$SCRATCH/gnn-scaling-128
DATA_DIR=data/kroneckers_strong_split/

conda activate $PROJECT_ROOT/../venv-dgl101

BATCH_SIZE=16384
N_HIDDEN=128

for path in $PROJECT_ROOT/$DATA_DIR/n1_*; do
    graph_name=${path##*/}
    for model in "va" "agnn" "gat"; do
        if [[ -f "$PROJECT_ROOT/$DATA_DIR/$graph_name/node_norm_${model}.json" ]]; then
            echo -e "################" 
            echo -e "# SKIPPING ${model} #" 
            echo -e "################" 
        else 
            echo -e "\n\nRUNNING $model"
            srun python $PROJECT_ROOT/src/train_norm.py \
                --graph_name $graph_name \
                --data_dir $DATA_DIR \
                --batch_size_eval $BATCH_SIZE \
                --num_hidden $N_HIDDEN \
                --n_classes $N_HIDDEN \
                --model $model
        fi
    done
done
