##!/bin/bash
#source params.sh
#
#for db in "${merge_bench[@]}"; do
#  if [ "$db" == "sift" ]; then
#    ef=200
#    M=16
#    sub_ef=100
#    sub_M=16
#  elif [ "$db" == "deep1M" ]; then
#    ef=200
#    M=32
#    sub_ef=100
#    sub_M=32
#  elif [ "$db" == "gist" ]; then
#    ef=200
#    M=64
#    sub_ef=100
#    sub_M=64
#  elif [ "$db" == "glove100d" ]; then
#    ef=200
#    M=64
#    sub_ef=100
#    sub_M=64
#  elif [ "$db" == "msong" ]; then
#    ef=200
#    M=32
#    sub_ef=100
#    sub_M=32
#  elif [ "$db" == "crawl" ]; then
#    ef=200
#    M=64
#    sub_ef=100
#    sub_M=64
#  else
#    echo "Unknown dataset: $db"
#    exit 1
#  fi
#
#  EXECUTABLE="/home/jlc/hnsw-merge/cmake-build-debug/test_RGTM_merge"
#  DATA_FILE="/home/jlc/hnsw-merge/data/${db}/random/multi-index-data/10parts/${db}_random_base.fvecs"
#  GRAPH_INDEX_FILE="/home/jlc/hnsw-merge/data/${db}/random/multi-index-merged/10parts/${db}_randomP"
#  MERGED_NSG_PATH="/home/jlc/hnsw-merge/data/${db}/random/multi-index-merged/10parts/${db}_random_RGTM"
#
#done
#
#
#ET=0
#RATIO=1.0
#
## Log file to store outputs
#LOG_FILE="/home/jlc/hnsw-merge/data/${db}/random/performance/10parts/RGTM_merge.log"
#
## Parameter combinations for G, L, and S
#PARAMS=(
#  "200 10 3"
#  "200 20 3"
#  "200 30 3"
#  "200 40 3"
#  "200 10 5"
#  "200 20 5"
#  "200 30 5"
#  "200 40 5"
#  "200 10 10"
#  "200 20 10"
#  "200 30 10"
#  "200 40 10"
#)
#
#morder="pairwise"
#
## Start logging
#echo "Starting index construction at $(date)" | tee -a "$LOG_FILE"
#
## Loop through each parameter combination and run the executable
#for param in "${PARAMS[@]}"; do
#  read -r G L S <<< "$param"
#
#  OUTPUT_FILE="${MERGED_NSG_PATH}_et${ET}_ef${G}_${L}_${S}_M32.hnsw"
#
#  $EXECUTABLE $DATA_FILE $G $L $S 32 40 16 10 $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder 2>&1 | tee -a "$LOG_FILE"
#done
#
## Finish logging
#echo "Index construction completed at $(date)" | tee -a "$LOG_FILE"