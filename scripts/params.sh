#!/bin/bash

CONDA_PATH="/home/jlc/miniconda3"
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate myenv

#sift : (1000000, 128); query : (10000, 128) 40
#deep1M : (1000000, 96); query : (1000, 96) 40
#msong : (990000, 420); query : (1000, 420) 30
#msmarco1M : (1000000, 1024); query : (1000, 1024) 50


# datasets=("msmarc10m" "deep10m" "anton10m" "imagenet10m")
datasets=("deep100M")
partition_methods=("random")
ms=(40)
