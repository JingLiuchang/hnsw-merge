#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
  if [ "$db" == "anton10m" ]; then
      ef=40
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
            "40 10 3"
              #"40 20 3"
              #"40 30 3"
            )
  elif [ "$db" == "imagenet10m" ]; then
        ef=50
        M=30
        sub_ef=300
        sub_M=30
        PARAMS=(
#                "50 10 3"
                #70 20 3"
                "50 10 3"
                #"50 30 3"
              )
  elif [ "$db" == "deep10m" ]; then
      ef=60
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
#              "60 10 3"
              #"60 40 3"
              "60 10 3"
            )
  elif [ "$db" == "msmarc10m" ]; then
      ef=50
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
#              "50 10 3"
              "50 10 3"
#              "50 30 3"
              #"50 30 3"
            )
  elif [ "$db" == "deep100M" ]; then
        ef=60
        M=30
        sub_ef=500
        sub_M=30
        PARAMS=(
                "60 10 3"
                "60 20 3"
                #"60 40 3"
              )
  else
    echo "Unknown dataset: $db"
    exit 1
  fi

  for m in "${ms[@]}"; do
    EXECUTABLE="${REPO_PATH}/cmake-build-debug/test_RGTM_merge"
    DATA_FILE="${DATA_PATH}/${db}/kmeans/multi-index-data/${m}parts/${db}_kmeans_base.fvecs"
    GRAPH_INDEX_FILE="${DATA_PATH}/${db}/kmeans/multi-index-merged/${m}parts/${db}_kmeansP"
    MERGED_NSG_PATH="${DATA_PATH}/${db}/kmeans/multi-index-merged/${m}parts/${db}_kmeans_RGTM"
    MERGE_ORDER_GRAPH="${DATA_PATH}/${db}/kmeans/multi-index-data/${m}parts/${db}_centroid_2hopgraph.fvecs"

    ET=0
    RATIO=1.0

    # Log file to store outputs
    LOG_FILE="${REPO_PATH}/performance/${db}/${m}parts/RGTM_merge.log"

    morder="weighted-graph"

    # Start logging
    echo "Starting index construction at $(date)" | tee -a "$LOG_FILE"

    # Loop through each parameter combination and run the executable
    for param in "${PARAMS[@]}"; do
      read -r G L S <<< "$param"

      OUTPUT_FILE="${MERGED_NSG_PATH}_${morder}_ef${G}_${L}_${S}_M${M}.hnsw"

      #echo "$EXECUTABLE $DATA_FILE $G $L $S ${M} ${sub_ef} ${sub_M} 2 $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder"
      $EXECUTABLE $DATA_FILE $G $L $S ${M} ${sub_ef} ${sub_M} $m $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder $MERGE_ORDER_GRAPH 2>&1 | tee -a "$LOG_FILE"
    done

    NGM_MORDER="pairwise"
    NGM_EXECUTABLE="${REPO_PATH}/cmake-build-debug/test_NGM_merge"
    NGM_MERGED_NSG_PATH="${DATA_PATH}/${db}/kmeans/multi-index-merged/${m}parts/${db}_kmeans_NGM"
    NGM_OUTPUT_FILE="${NGM_MERGED_NSG_PATH}_${NGM_MORDER}_ef${ef}_M${M}.hnsw"

    $NGM_EXECUTABLE $DATA_FILE $ef ${M} ${sub_ef} ${sub_M} $m $GRAPH_INDEX_FILE $NGM_OUTPUT_FILE $ET $RATIO $NGM_MORDER 2>&1 | tee -a "$LOG_FILE"

    # Finish logging
    echo "Index construction completed at $(date); m=${m}" | tee -a "$LOG_FILE"

  done
done