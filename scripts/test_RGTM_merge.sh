#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
  if [ "$db" == "sift" ]; then
    ef=40
    M=25
    sub_ef=200
    sub_M=25
    PARAMS=(
        "40 5 3"
        "40 10 3"
        "40 20 3"
        "40 30 3"
      )
  elif [ "$db" == "deep1M" ]; then
    ef=40
    M=30
    sub_ef=200
    sub_M=30
    PARAMS=(
            "40 5 3"
            "40 10 3"
            "40 20 3"
            "40 30 3"
          )
  elif [ "$db" == "gist" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  elif [ "$db" == "glove100d" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  elif [ "$db" == "msong" ]; then
    ef=30
    M=40
    sub_ef=200
    sub_M=40
    PARAMS=(
            "30 5 3"
            "30 10 3"
            "30 15 3"
            "30 20 3"
          )
  elif [ "$db" == "crawl" ]; then
    ef=200
    M=35
    sub_ef=200
    sub_M=35
  elif [ "$db" == "msmarco1M" ]; then
    ef=50
    M=30
    sub_ef=200
    sub_M=30
    PARAMS=(
            "50 5 3"
            "50 10 3"
            "50 20 3"
            "50 40 3"
          )
#  elif [ "$db" == "anton10m" ]; then
#      ef=50
#      M=30
#      sub_ef=300
#      sub_M=30
#      PARAMS=(
#              "50 5 3"
#              "50 10 3"
#              "50 20 3"
#              "50 40 3"
#            )
  elif [ "$db" == "anton10m" ]; then
      ef=40
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
              "40 20 3"
              "40 30 3"
            )
  elif [ "$db" == "imagenet10m" ]; then
        ef=50
        M=30
        sub_ef=300
        sub_M=30
        PARAMS=(
                "50 10 3"
                "50 20 3"
                "50 40 3"
              )
  elif [ "$db" == "deep10m" ]; then
      ef=50
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
#              "50 10 3"
#              "50 20 3"
              "50 40 3"
            )
  elif [ "$db" == "msmarc10m" ]; then
      ef=50
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
              "50 10 3"
              "50 20 3"
              "50 30 3"
              "50 40 3"
            )
  elif [ "$db" == "deep100M" ]; then
      ef=60
      M=30
      sub_ef=500
      sub_M=30
      PARAMS=(
#              "60 10 3"
              "60 20 3"
              #"60 40 3"
            )
  else
    echo "Unknown dataset: $db"
    exit 1
  fi

  for m in "${ms[@]}"; do
    EXECUTABLE="${REPO_PATH}/cmake-build-debug/test_RGTM_merge"
    DATA_FILE="${DATA_PATH}/${db}/random/multi-index-data/${m}parts/${db}_random_base.fvecs"
    GRAPH_INDEX_FILE="${DATA_PATH}/${db}/random/multi-index-merged/${m}parts/${db}_randomP"
    MERGED_NSG_PATH="${DATA_PATH}/${db}/random/multi-index-merged/${m}parts/${db}_random_RGTM"

    ET=0
    RATIO=1.0

    # Log file to store outputs
    LOG_FILE="${REPO_PATH}/performance/${db}/${m}parts/RGTM_merge.log"

    morder="unweighted-graph"

    # Start logging
    echo "Starting index construction at $(date)" | tee -a "$LOG_FILE"

    # Loop through each parameter combination and run the executable
    for param in "${PARAMS[@]}"; do
      read -r G L S <<< "$param"

      OUTPUT_FILE="${MERGED_NSG_PATH}_${morder}_ef${G}_${L}_${S}_M${M}.hnsw"

      echo "$EXECUTABLE $DATA_FILE $G $L $S ${M} ${sub_ef} ${sub_M} $m $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder"
      $EXECUTABLE $DATA_FILE $G $L $S ${M} ${sub_ef} ${sub_M} $m $GRAPH_INDEX_FILE $OUTPUT_FILE $ET $RATIO $morder 2>&1 | tee -a "$LOG_FILE"
    done

    NGM_MORDER="path"
    NGM_EXECUTABLE="${REPO_PATH}/cmake-build-debug/test_NGM_merge"
    NGM_MERGED_NSG_PATH="${DATA_PATH}/${db}/random/multi-index-merged/${m}parts/${db}_random_NGM"
    NGM_OUTPUT_FILE="${NGM_MERGED_NSG_PATH}_${NGM_MORDER}_ef${ef}_M${M}.hnsw"

    $NGM_EXECUTABLE $DATA_FILE $ef ${M} ${sub_ef} ${sub_M} $m $GRAPH_INDEX_FILE $NGM_OUTPUT_FILE $ET $RATIO $NGM_MORDER 2>&1 | tee -a "$LOG_FILE"

    # Finish logging
    echo "Index construction completed at $(date); m=${m}" | tee -a "$LOG_FILE"

  done
done