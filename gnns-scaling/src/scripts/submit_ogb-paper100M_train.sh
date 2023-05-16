#!/bin/bash -l
#SBATCH --job-name="dgl-test"
#SBATCH --account="g34"
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

#SBATCH --output=dgl-test.txt
#SBATCH --error=dgl-test.txt

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
	--part_config data/ogb-paper100M/ogb-paper100M.json \
	--ip_config ip_config.txt \
	"$CONDA_PREFIX/bin/python $PROJECT_ROOT/src/train_dist.py --graph_name ogb-paper100M --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --n_classes 172 --num_gpus 1 --model va"
