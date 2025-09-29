#!/bin/bash

CONDA_PATH="/home/jlc/miniconda3"
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate myenv

datasets=("sift")
partition_methods=("kmeans" "random" "overlap")
ms=(2 10 100)

echo "Activated conda environment: "
which python