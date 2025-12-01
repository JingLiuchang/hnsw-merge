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
    EXECUTABLE="${REPO_PATH}/cmake-build-debug/test_hnsw_search"
    DATA_FILE="${DATA_PATH}/${db}/kmeans/multi-index-data/${m}parts/${db}_kmeans_base.fvecs"
    QUERY_FILE="${DATA_PATH}/${db}/${db}_query.fvecs"
    GT_FILE="${DATA_PATH}/${db}/kmeans/multi-index-data/${m}parts/${db}_kmeans_groundtruth.ivecs"
    OUTPUT_PATH="${REPO_PATH}/performance/${db}/${m}parts"
    MERGED_NSG_PATH="${DATA_PATH}/${db}/kmeans/multi-index-merged/${m}parts/${db}_kmeans_RGTM"

    # Fixed parameters for the search
    K=10          # Number of nearest neighbors to retrieve
    MIN_EF=10     # Minimum ef value
    MAX_EF=150    # Maximum ef value
    STEPSIZE=10   # Step size for increasing ef
    morder="weighted-graph"

    # Loop through each parameter combination
    for param in "${PARAMS[@]}"; do
      # Parse the parameter string (G_L_S)
      read -r G L S <<< "$param"

      # Construct the input HNSW graph file name and output CSV file name
      GRAPH_INDEX_FILE="${MERGED_NSG_PATH}_${morder}_ef${G}_${L}_${S}_M${M}.hnsw"
      PERFORMANCE_CSV="${OUTPUT_PATH}/${db}_kmeans_RGTM_${morder}_ef${G}_${L}_${S}_M${M}_K${K}.csv"

      # Run the search command
      # echo "Running: $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $PERFORMANCE_CSV"
      $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $PERFORMANCE_CSV
    done

    NGM_GRAPH_INDEX_FILE="${DATA_PATH}/${db}/kmeans/multi-index-merged/${m}parts/${db}_kmeans_NGM_pairwise_ef${ef}_M${M}.hnsw"
    NGM_PERFORMANCE_CSV="${OUTPUT_PATH}/${db}_kmeans_NGM_pairwise_ef${ef}_M${M}_K${K}.csv"
    $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $NGM_GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $NGM_PERFORMANCE_CSV
  done
done

echo "All tests completed."