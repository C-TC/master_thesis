#!/bin/bash -l
#SBATCH --job-name="synth-graph-partitioning"
#SBATCH --account="g34"
#SBATCH --nodes=1
#SBATCH --mem=1T
#SBATCH --time=3:00:00
#SBATCH --partition=fat
#SBATCH --output=graph_gen.txt
#SBATCH --error=graph_gen.txt

# ault does not have scratch
PROJECT_ROOT=$HOME/gnns-scaling
cd $PROJECT_ROOT

conda activate $PROJECT_ROOT/venv

#PARAMS=("4;0.0001" "4;0.01" "16;0.0001" "16;0.01")
#PARAMS=("1;0.0001" "1;0.01" "8;0.01" "32;0.0001")
PARAMS=("32;0.01")

for params in ${PARAMS[@]}; do 
	IFS=";" read -r -a params <<< "${params}"
	node=${params[0]}
	sparsity=${params[1]}

	dataset="n"$node"_a17_s"$sparsity
	cmd1=$(printf "python $PROJECT_ROOT/src/scripts/partition_graph.py 
		--balance_edges
		--output data/$dataset
		--dataset $dataset 
		--num_parts $node")

	[[ ! -z $cmd ]] && cmd+=" && "
	cmd+=$cmd1
done
cmd="srun "$cmd
echo -e $cmd
eval $(echo -e $cmd)
