#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="${REPO_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_PATH="${DATA_PATH:-/home/bld/datasets/merge_bench}"
BUILD_DIR="${BUILD_DIR:-/tmp/hnsw-merge-parallelism-build}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export REPO_PATH DATA_PATH BUILD_DIR PYTHON_BIN

#sift : (1000000, 128); query : (10000, 128) 40
#deep1M : (1000000, 96); query : (1000, 96) 40
#glove100d : (1000000, 100); query : (1000, 100)
#crawl : (1000000, 300); query : (10000, 300)
#msong : (990000, 420); query : (1000, 420) 30
#gist : (1000000, 960); query : (1000, 960)
#msmarco1M : (1000000, 1024); query : (1000, 1024) 50



#datasets=("sift" "deep1M" "gist" "msmarco1M" "anton1m" "imagenet1m")
datasets=("imagenet1m")
partition_methods=("random")
ms=(2)
Ts=(72)
