#!/bin/bash

# Define the executable and input/output paths
#EXECUTABLE="/home/jlc/hnswlib/cmake-build-debug/test_RGTM_merge"
#DATA_FILE="/home/jlc/hnswlib/data/deep10M/random/bi-index-data/deep10M_random_base.fvecs"
#GRAPH_INDEX_FILE="/home/jlc/hnswlib/data/deep10M/random/bi-index-merged/deep10M_randomP"
#MERGED_NSG_PATH="/home/jlc/hnswlib/data/deep10M/random/bi-index-merged/deep10M_random_RGTM"

EXECUTABLE="/home/jlc/hnswlib/cmake-build-debug/test_RGTM_merge"
DATA_FILE="/home/jlc/hnswlib/data/deep10M/random/multi-index-data/10parts/deep10M_random_base.fvecs"
GRAPH_INDEX_FILE="/home/jlc/hnswlib/data/deep10M/random/multi-index-merged/10parts/deep10M_randomP"
MERGED_NSG_PATH="/home/jlc/hnswlib/data/deep10M/random/multi-index-merged/10parts/deep10M_random_RGTM"

ET=0
RATIO=1.0

# Log file to store outputs
LOG_FILE="/home/jlc/hnswlib/data/deep10M/random/performance/10parts/RGTM_merge.log"

# Parameter combinations for G, L, and S
PARAMS=(
  "80 10 3"
  "80 20 3"
  "80 30 3"
  "80 40 3"
  "80 10 5"
  "80 20 5"
  "80 30 5"
  "80 40 5"
  "80 20 10"
  "80 30 10"
  "80 40 3"
  "40 20 3"
  "40 20 5"
  "40 20 10"
  "20 10 3"
  "20 10 5"
  "20 5 3"
)

morder="pairwise"

# Start logging
echo "Starting index construction at $(date)" | tee -a "$LOG_FILE"

# Loop through each parameter combination and run the executable
for param in "${PARAMS[@]}"; do
  read -r G L S <<< "$param"

  OUTPUT_FILE="${MERGED_NSG_PATH}_et${ET}_ef${G}_${L}_${S}_M32.hnsw"

  $EXECUTABLE $DATA_FILE $G $L $S 32 40 16 10 $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder 2>&1 | tee -a "$LOG_FILE"
done

# Finish logging
echo "Index construction completed at $(date)" | tee -a "$LOG_FILE"