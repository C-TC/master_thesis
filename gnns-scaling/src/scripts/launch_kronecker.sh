# NODESs=( 1 2 4 8 16 32 64 128 256 512 )
NODESs=( 128 )
# MODELs=( cgnn vanilla agnn gat )
MODELs=( vanilla )
ARCH=(gpu)
# RESULT_FOLDER=${1}
FILE=${1}

function create_launch {


  script_body="#!/bin/bash
#SBATCH --job-name=${MODEL}_${NODES}_${ARCH}_kronecker
#SBATCH --account=g34
#SBATCH --nodes=${NODES}
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00

# module load daint-gpu
# module load gcc/9.3.0
# module load intel-oneapi/2022.1.0
# module swap PrgEnv-cray/6.0.10 PrgEnv-gnu

# Setup the compiler
#
export CC=`which cc`
export CXX=`which CC`

# Enable dynamic linking
#
export CRAYPE_LINK_TYPE=dynamic

#some important stuff
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_NEMESIS_ASYNC_PROGRESS=MC
export MPICH_RDMA_ENABLED_CUDA=1

# Enable threading
#
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

DACE_compiler_use_cache=1 LD_PRELOAD=/usr/lib64/libcuda.so.1:/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.0 srun --time=5 -N ${NODES} python samples/distributed/kronecker_${ARCH}.py -f ${FILE} -m ${MODEL}"

      
  script_name=___tmp_script_weak_${ARCH}_${MODEL}__${NODES}___

#   script_folder=${RESULT_FOLDER}/${MODEL}__${NODES}__${ARCH}__weak_exec_dir

#   if [[ -d ${script_folder} ]]; then
    # echo "experiment exists already"
#   else

    # mkdir ${script_folder}

    # echo "${script_body}" > ${script_folder}/${script_name}.sbatch
    echo "${script_body}" > ${script_name}.sbatch

    # cd ${script_folder}

    sbatch ${script_name}.sbatch
    
    # cd ..
#   fi
}

for M in ${MODELs[@]}; do
  export MODEL=${M}
  for N in ${NODESs[@]}; do
    export NODES=${N}
    echo "======= Processing ${M} for ${N} nodes on ${FILE} graph"
    create_launch
  done
done
