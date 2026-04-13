#!/bin/bash
source params.sh

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
    elif [ "$db" == "msmarc10m" ]; then
      ef=50
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
              "50 10 3"
            )
    elif [ "$db" == "imagenet10m" ]; then
      ef=50
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
              "50 10 3"
            )
    elif [ "$db" == "anton10m" ]; then
      ef=40
      M=30
      sub_ef=300
      sub_M=30
      PARAMS=(
            "40 10 3"
            )
    else
      echo "Unknown dataset: $db"
      exit 1
    fi

  EXECUTABLE="/home/jlc/hnsw-merge/cmake-build-debug/test_hnsw_search"
  DATA_FILE="/mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_base.fvecs"
  QUERY_FILE="/mnt/ssd/merge_bench/${db}/${db}_query.fvecs"
  GT_FILE="/mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_groundtruth.ivecs"
  OUTPUT_PATH="/mnt/ssd/merge_bench/${db}/random/performance/bi"
  MERGED_NSG_PATH="/mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_random_RGTM"

  # Fixed parameters for the search
  K=10          # Number of nearest neighbors to retrieve
  MIN_EF=$K      # Minimum ef value
  MAX_EF=$((K + 150))  # Maximum ef value
  STEPSIZE=10    # Step size for increasing ef
  mkdir -p "${OUTPUT_PATH}/K${K}"

  # Loop through each parameter combination
  for param in "${PARAMS[@]}"; do
    # Parse the parameter string (G_L_S)
    read -r G L S <<< "$param"

    # Construct the input HNSW graph file name and output CSV file name
    GRAPH_INDEX_FILE="${MERGED_NSG_PATH}_et0_ef${G}_${L}_${S}_M${M}.hnsw"
    PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/${db}_random_RGTM_et0_ef${G}_${L}_${S}_M${M}_K${K}.csv"
    mkdir -p "${OUTPUT_PATH}/K${K}/"
#    PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/tmp.csv"

    # Run the search command
    # echo "Running: $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $PERFORMANCE_CSV"
    $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $PERFORMANCE_CSV
#    mkdir -p "/home/jlc/pg-fast-merging/performance/${db}/"
#    cp $PERFORMANCE_CSV "/home/jlc/pg-fast-merging/performance/${db}/"
  done

#  NGM_GRAPH_INDEX_FILE="/mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_random_NGM_et0_ef${ef}_M${M}.hnsw"
#  NGM_PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/${db}_random_NGM_et0_ef${ef}_M${M}_K${K}.csv"
#  $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $NGM_GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $NGM_PERFORMANCE_CSV
#  mkdir -p "/home/jlc/pg-fast-merging/performance/${db}/"
#  cp $NGM_PERFORMANCE_CSV "/home/jlc/pg-fast-merging/performance/${db}/"
done

echo "All tests completed."