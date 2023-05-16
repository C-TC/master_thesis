#!/bin/bash
search_dir=$SCRATCH/gnns-scaling/src/scripts/submit_exp/
for entry in "$search_dir"/*
do
  sbatch "$entry"
done
