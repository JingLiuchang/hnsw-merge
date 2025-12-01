#!/bin/bash

# set your python conda environment
CONDA_PATH="/home/jlc/miniconda3"
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate myenv

REPO_PATH="/home/jlc/hnsw-merge"
DATA_PATH="/mnt/ssd/merge_bench"
datasets=("deep10m" "msmarc10m" "anton10m" "imagenet10m")
partition_methods=("random") # random / kmeans / overlapping
ms=(8) # number of partitions
