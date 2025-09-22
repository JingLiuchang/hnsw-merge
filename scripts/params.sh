#!/bin/bash

CONDA_PATH="/home/jlc/miniconda3"
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate myenv
datasets=("deep1M")
partition_methods=("kmeans" "random")

echo "Activated conda environment: "
which python