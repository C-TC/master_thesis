#!/usr/bin/env bash

BATCH_SIZE=16384
N_HIDDEN=128
EXP_TYPE=weak
time_limit="1:30:00"

DATA_DIR="data/kroneckers_${EXP_TYPE}_split"
[[ -z ${PROJECT_ROOT} ]] && PROJECT_ROOT=$SCRATCH/gnn-scaling-128
OUTDIR="$PROJECT_ROOT"/submit_kronecker_${EXP_TYPE}
mkdir $OUTDIR
echo "CREATING:"

for path in ${PROJECT_ROOT}/${DATA_DIR}/n*; do
    filename=${path##*/}
    graph_info=${EXP_TYPE}_${path##*/}

    IFS="_" read -r -a CHUNKS <<< "${filename}"
    node=${CHUNKS[0]}
    node=${node:1}

read -r -d '' cmd << EOM
#!/bin/bash -l
#SBATCH --job-name="${graph_info}-dglDist_exp"
#SBATCH --account="g34"
#SBATCH --time=${time_limit}
#SBATCH --nodes=${node}
#SBATCH --partition=normal
#SBATCH --constraint=gpu

#SBATCH --output=exp_${graph_info}.log
#SBATCH --error=exp_${graph_info}.log

module load daint-gpu

PROJECT_ROOT=$PROJECT_ROOT
conda activate $SCRATCH/venv-dgl091

EOM


read -r -d '' RUN_MODEL_CMD << EOM
echo chaning to $path
cd $path

srun python \$PROJECT_ROOT/src/scripts/create_node_list.py &&\n
for model in "va" "agnn" "gat"; do
    if [[ -f "$path/node00_\${model}.json" ]]; then
        echo -e "###################" 
        echo -e "# SKIPPING \${model} #" 
        echo -e "###################" 
    else 
        echo -e "\\\n\\\nRUNNING \$model"
        python \$PROJECT_ROOT/src/launch.py \\
            --workspace \$PROJECT_ROOT \\
            --ssh_username \$USER \\
            --num_trainers 1 \\
            --num_samplers 1 \\
            --num_servers 1 \\
            --part_config $path/${filename}.json \\
            --ip_config $path/ip_config.txt \\
            "\$CONDA_PREFIX/bin/python \$PROJECT_ROOT/src/train_dist.py \\\\
                --graph_name ${filename} \\\\
                --ip_config $path/ip_config.txt \\\\
                --batch_size $BATCH_SIZE \\\\
                --data_dir $DATA_DIR \\\\
                --num_hidden $N_HIDDEN \\\\
                --n_classes $N_HIDDEN \\\\
                --num_layers 3 \\\\
                --num_gpus 1 \\\\
                --model \$model \\\\
                --num_epochs 5"
    fi
done
EOM

    outpath="$OUTDIR/$filename".sh
    echo ${outpath}
    echo -e "$cmd""$RUN_MODEL_CMD" > "${outpath}"
done

exit
