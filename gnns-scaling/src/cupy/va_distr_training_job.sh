#!/bin/bash -l
#SBATCH --job-name="distr_va"
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

srun python va_training_distr.py --layers 3 --features 128 -d kronecker -v 2097152 -e 439804651