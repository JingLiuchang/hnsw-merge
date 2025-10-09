#!/bin/bash

CONDA_PATH="/home/jlc/miniconda3"
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate myenv

datasets=("deep10M")
partition_methods=("kmeans" "random" "overlap")
ms=(10 20)

echo "Activated conda environment: "
which python