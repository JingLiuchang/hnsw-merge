#!/bin/bash

# Define the executable and fixed input/output paths
#EXECUTABLE="/home/jlc/hnswlib/cmake-build-debug/test_hnsw_search"
#DATA_FILE="/home/jlc/hnswlib/data/deep1M/kmeans/bi-index-data/deep1M_kmeans_base.fvecs"
#QUERY_FILE="/home/jlc/hnswlib/data/deep1M/deep1M_query.fvecs"
#GT_FILE="/home/jlc/hnswlib/data/deep1M/kmeans/bi-index-data/deep1M_kmeans_groundtruth.ivecs"
#OUTPUT_PATH="/home/jlc/hnswlib/data/deep1M/kmeans/performance/bi"

EXECUTABLE="/home/jlc/hnswlib/cmake-build-debug/test_hnsw_search"
DATA_FILE="/home/jlc/hnswlib/data/deep1M/kmeans/multi-index-data/5parts/deep1M_kmeans_base.fvecs"
QUERY_FILE="/home/jlc/hnswlib/data/deep1M/deep1M_query.fvecs"
GT_FILE="/home/jlc/hnswlib/data/deep1M/kmeans/multi-index-data/5parts/deep1M_kmeans_groundtruth.ivecs"
OUTPUT_PATH="/home/jlc/hnswlib/data/deep1M/kmeans/performance/5parts"

# Parameter combinations for ef80_G_L_S
PARAMS=(
  "80_10_3"
  "80_20_3"
  "80_30_3"
  "80_40_3"
  "80_10_5"
  "80_20_5"
  "80_30_5"
  "80_40_5"
  "80_20_10"
  "80_30_10"
  "80_40_3"
  "40_20_3"
  "40_20_5"
  "40_20_10"
  "20_10_3"
  "20_10_5"
  "20_5_3"
)

# Fixed parameters for the search
K=10          # Number of nearest neighbors to retrieve
MIN_EF=10     # Minimum ef value
MAX_EF=150    # Maximum ef value
STEPSIZE=10   # Step size for increasing ef

# Loop through each parameter combination
for param in "${PARAMS[@]}"; do
  # Parse the parameter string (G_L_S)
  IFS="_" read -r G L S <<< "$param"

  # Construct the input HNSW graph file name and output CSV file name
  GRAPH_INDEX_FILE="/home/jlc/hnswlib/data/deep1M/kmeans/multi-index-merged/5parts/deep1M_kmeans_RGTM_et0_ef${G}_${L}_${S}_M32.hnsw"
  PERFORMANCE_CSV="${OUTPUT_PATH}/deep1M_kmeans_RGTM_et0_ef${G}_${L}_${S}_M32.csv"

  # Run the search command
  echo "Running: $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $PERFORMANCE_CSV"
  $EXECUTABLE $DATA_FILE $QUERY_FILE $GT_FILE $GRAPH_INDEX_FILE $K $MIN_EF $MAX_EF $STEPSIZE $PERFORMANCE_CSV
done

echo "All tests completed."