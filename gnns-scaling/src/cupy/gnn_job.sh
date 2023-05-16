#!/usr/bin/env bash

DATA_DIR="data/kroneckers_split"
[[ -z ${PROJECT_ROOT} ]] && PROJECT_ROOT=$SCRATCH/gnns-scaling
OUTDIR="$PROJECT_ROOT"/submit_kronecker_cupy
mkdir $OUTDIR
echo "CREATING:"

NODES=(1 4 16 64 256)
PARAMS=("131072_171798692" "262144_687194767" "1048576_109951163" "2097152_439804651")


for model in "va" "gat" "agnn"; do 
    for k in "16" "128"; do 
        for task in "training" "inference"; do
            for nodes in "${NODES[@]}"; do
                for params in "${PARAMS[@]}"; do

IFS=_ read -r vertices edges <<< $params

filename=${model}_${task}_n${nodes}_v${vertices}_e${edges}_k${k}
if [[ "$model" == "gat" ]]; then
    cmd="gat_distr_bench.py --repeat 5 --warmup 1"
    if [[ "$task" == "inference" ]]; then
        cmd=$cmd" --inference"
    fi
else
    cmd=${model}_${task}_distr.py
fi

cmd="$PROJECT_ROOT/src/cupy/"$cmd

read -r -d '' cmd << EOM
#!/bin/bash -l
#SBATCH --job-name="${filename}"
#SBATCH --account="g34"
#SBATCH --time=1:00:00
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=exp_${filename}.log
#SBATCH --error=exp_${filename}.log

module load daint-gpu
module load cudatoolkit
module load numpy
module load daint-gpu

cd ${PROJECT_ROOT}/src/cupy

srun python ${cmd} -d kronecker -v ${vertices} -e ${edges} --layers 3 --features ${k}
EOM

outpath=$OUTDIR/"$filename".sh
echo ${outpath}
echo -e "$cmd" > "${outpath}"
                done
            done
        done
    done
done
