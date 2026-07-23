#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/params.sh"

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

  EXECUTABLE="${BUILD_DIR}/test_hnsw_search"
  DATA_FILE="${DATA_PATH}/${db}/random/bi-index-data/${db}_random_base.fvecs"
  QUERY_FILE="${DATA_PATH}/${db}/${db}_query.fvecs"
  GT_FILE="${DATA_PATH}/${db}/random/bi-index-data/${db}_random_groundtruth.ivecs"
  OUTPUT_PATH="${DATA_PATH}/${db}/random/performance/bi"
  INDEX_PATH="${DATA_PATH}/${db}/random/bi-index-merged"
  MERGED_NSG_PATH="${INDEX_PATH}/${db}_random_RGTM"

  # Fixed parameters for the search
  K=10          # Number of nearest neighbors to retrieve
  MIN_EF=$K     # Minimum ef value
  MAX_EF=$((K + 150))  # Maximum ef value
  STEPSIZE=5    # Step size for increasing ef
  mkdir -p "${OUTPUT_PATH}/K${K}"

  run_search_sweeps() {
    local method=$1
    local index_file=$2
    local qps_csv=$3
    local ndc_csv=$4

    echo "Running ${method} QPS sweep without metric instrumentation"
    "$EXECUTABLE" "$DATA_FILE" "$QUERY_FILE" "$GT_FILE" "$index_file" \
      "$K" "$MIN_EF" "$MAX_EF" "$STEPSIZE" "$qps_csv"

    echo "Running ${method} NDC sweep"
    "$EXECUTABLE" "$DATA_FILE" "$QUERY_FILE" "$GT_FILE" "$index_file" \
      "$K" "$MIN_EF" "$MAX_EF" "$STEPSIZE" /dev/null "$ndc_csv"
  }

#   # Loop through each parameter combination
#   for param in "${PARAMS[@]}"; do
#     # Parse the parameter string (G_L_S)
#     read -r G L S <<< "$param"

#     # Construct the input HNSW graph file name and output CSV file name
#     GRAPH_INDEX_FILE="${MERGED_NSG_PATH}_et0_ef${G}_${L}_${S}_M${M}_adjacent-blocks.hnsw"
#     PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/${db}_random_RGTM_et0_ef${G}_${L}_${S}_M${M}_K${K}.csv"
#     NDC_PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/NDC_${db}_random_RGTM_et0_ef${G}_${L}_${S}_M${M}_K${K}.csv"
#     mkdir -p "${OUTPUT_PATH}/K${K}/"
# #    PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/tmp.csv"

#     run_search_sweeps "RGTM" "$GRAPH_INDEX_FILE" "$PERFORMANCE_CSV" "$NDC_PERFORMANCE_CSV"
# #    mkdir -p "/home/jlc/pg-fast-merging/performance/${db}/"
# #    cp $PERFORMANCE_CSV "/home/jlc/pg-fast-merging/performance/${db}/"
#   done

#   NGM_GRAPH_INDEX_FILE="${INDEX_PATH}/${db}_random_NGM_et0_ef${ef}_M${M}.hnsw"
#   NGM_PERFORMANCE_CSV="${OUTPUT_PATH}/K${K}/${db}_random_NGM_et0_ef${ef}_M${M}_K${K}.csv"
#   NGM_NDC_CSV="${OUTPUT_PATH}/K${K}/NDC_${db}_random_NGM_et0_ef${ef}_M${M}_K${K}.csv"
#   run_search_sweeps "NGM" "$NGM_GRAPH_INDEX_FILE" "$NGM_PERFORMANCE_CSV" "$NGM_NDC_CSV"

  BUILD_AS_ONE_INDEX="${INDEX_PATH}/${db}_random_BuildAsOne_ef200_M${M}.hnsw"
  BUILD_AS_ONE_CSV="${OUTPUT_PATH}/K${K}/${db}_random_BuildAsOne_ef200_M${M}_K${K}.csv"
  BUILD_AS_ONE_NDC_CSV="${OUTPUT_PATH}/K${K}/NDC_${db}_random_BuildAsOne_ef200_M${M}_K${K}.csv"
  run_search_sweeps "BuildAsOne" "$BUILD_AS_ONE_INDEX" "$BUILD_AS_ONE_CSV" "$BUILD_AS_ONE_NDC_CSV"
done

echo "All tests completed."
