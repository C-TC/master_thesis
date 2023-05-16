#!/bin/bash -l
#SBATCH --job-name="reddit-debug-dist"
#SBATCH --account="g34"
#SBATCH --time=00:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu

#SBATCH --output=reddit_debug_dist.log
#SBATCH --error=reddit_debug_dist.log

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

module load daint-gpu

PROJECT_ROOT=$SCRATCH/gnns-scaling

conda activate $PROJECT_ROOT/venv

srun python $PROJECT_ROOT/src/scripts/create_node_list.py && \
	python $PROJECT_ROOT/src/launch.py \
	--workspace $PROJECT_ROOT \
	--ssh_username $USER \
	--num_trainers 1 \
	--num_servers 1 \
	--part_config data/reddit/reddit.json \
	--ip_config ip_config.txt \
	"$CONDA_PREFIX/bin/python $PROJECT_ROOT/src/train_dist.py --graph_name reddit --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --n_classes 41 --num_gpus 1 --infer --model gat"
