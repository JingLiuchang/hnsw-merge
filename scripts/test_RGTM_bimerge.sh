#!/bin/bash
source params.sh

for T in "${Ts[@]}"; do # 并行测试

  for db in "${datasets[@]}"; do
    if [ "$db" == "sift" ]; then
      ef=40
      M=25
      sub_ef=200
      sub_M=25
      PARAMS=(
          "40 20 3"
        )
    elif [ "$db" == "deep1M" ]; then
      ef=40
      M=30
      sub_ef=200
      sub_M=30
      PARAMS=(
              "40 10 3"
            )
    elif [ "$db" == "msong" ]; then
      ef=30
      M=40
      sub_ef=200
      sub_M=40
      PARAMS=(
              "30 10 3"
            )
    elif [ "$db" == "msmarco1M" ]; then
      ef=50
      M=30
      sub_ef=200
      sub_M=30
      PARAMS=(
              "50 10 3"
            )
    elif [ "$db" == "anton1m" ]; then
      ef=40
      M=30
      sub_ef=200
      sub_M=30
      PARAMS=(
              "40 20 3"
            )
    elif [ "$db" == "imagenet1m" ]; then
      ef=40
      M=30
      sub_ef=200
      sub_M=30
      PARAMS=(
              "40 10 3"
            )
    elif [ "$db" == "gist" ]; then
      ef=40
      M=30
      sub_ef=200
      sub_M=30
      PARAMS=(
        "40 10 3"
      )
    elif [ "$db" == "deep10m" ]; then
      ef=60
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
              "60 30 3"
            )
    else
      echo "Unknown dataset: $db"
      exit 1
    fi

    EXECUTABLE="/home/jlc/hnsw-merge/cmake-build-debug/test_RGTM_merge"
    DATA_FILE="/mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_base.fvecs"
    GRAPH_INDEX_FILE="/mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_randomP"
    MERGED_NSG_PATH="/mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_random_RGTM"

    ET=0
    RATIO=1.0

    # Log file to store outputs
    # LOG_FILE="/mnt/ssd/merge_bench/${db}/random/logs/RGTM_merge.log"
  #  mkdir -p "/home/jlc/pg-fast-merging/performance/${db}/"
  #  LOG_FILE="/home/jlc/pg-fast-merging/performance/${db}/RGTM_merge.log"
    mkdir -p "/home/jlc/hnsw-merge/logs/${db}/"
    LOG_FILE="/home/jlc/hnsw-merge/logs/${db}/RGTM_merge.log"

    morder="pairwise"

    # Start logging
    echo "Starting index construction at $(date)" | tee -a "$LOG_FILE"

    # Loop through each parameter combination and run the executable
    for param in "${PARAMS[@]}"; do
      read -r G L S <<< "$param"

      OUTPUT_FILE="${MERGED_NSG_PATH}_et${ET}_ef${G}_${L}_${S}_M${M}.hnsw"

      echo "$EXECUTABLE $DATA_FILE $G $L $S ${M} ${sub_ef} ${sub_M} 2 $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder $T"
      $EXECUTABLE $DATA_FILE $G $L $S ${M} ${sub_ef} ${sub_M} 2 $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder $T 2>&1 | tee -a "$LOG_FILE"
    done

    NGM_EXECUTABLE="/home/jlc/hnsw-merge/cmake-build-debug/test_NGM_merge"
    NGM_MERGED_NSG_PATH="/mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_random_NGM"
    NGM_OUTPUT_FILE="${NGM_MERGED_NSG_PATH}_et${ET}_ef${ef}_M${M}.hnsw"

    #echo "$NGM_EXECUTABLE $DATA_FILE $ef ${M} ${sub_ef} ${sub_M} 2 $GRAPH_INDEX_FILE $NGM_OUTPUT_FILE $ET $RATIO $morder"
  #  $NGM_EXECUTABLE $DATA_FILE $ef ${M} ${sub_ef} ${sub_M} 2 $GRAPH_INDEX_FILE $NGM_OUTPUT_FILE $ET $RATIO $morder 2>&1 | tee -a "$LOG_FILE"

    # Finish logging
    echo "Index construction completed at $(date)" | tee -a "$LOG_FILE"

  #  cp "$LOG_FILE" "/home/jlc/pg-fast-merging/performance/${db}/"
  done

done