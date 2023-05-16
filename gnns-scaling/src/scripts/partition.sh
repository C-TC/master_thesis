#!/bin/bash -l
#SBATCH --job-name="gnn"
#SBATCH --account="g34"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120GB
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# dataset=n256_a17_e171798692_s0.01
# dataset=n256_a17_e17179869_s0.001
# dataset=n256_a17_e1717987_s0.0001
# dataset=n256_a18_e687194767_s0.01
# dataset=n256_a18_e68719477_s0.001
# dataset=n256_a18_e6871948_s0.0001
# dataset=n256_a19_e274877907_s0.001
# dataset=n256_a19_e27487791_s0.0001

conda activate gnn-scale

srun python partition_graph.py --balance_edges --output /users/ctianche/scratch/gnns-scaling/kroneckers_weak_raw/256/$dataset --dataset $dataset --num_parts 256
