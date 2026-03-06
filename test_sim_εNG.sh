#!/bin/bash
# Test script for SIM merge with εNG improvement

echo "=== SIM Merge εNG Improvement Test ==="
echo ""

# Check if test_SIM_merge exists
if [ ! -f "./test_SIM_merge" ]; then
    echo "Error: test_SIM_merge not found. Please run 'make test_SIM_merge' first."
    exit 1
fi

# Example parameters (adjust based on your dataset)
DATA_FILE="data/sift_base.fvecs"
GLOBAL_EF=200
LOCAL_EF=100
M=30
SUB_EF=200
SUB_M=25
GRAPH_NUM=4
GRAPH_INDEX_FILE="graphs/sift_T4_P"
MERGED_NSG_PATH="merged_sift_sim.hnsw"
MERGE_ORDER="pairwise"
THREADS=8

echo "Parameters:"
echo "  Data file: $DATA_FILE"
echo "  Global EF: $GLOBAL_EF"
echo "  Local EF: $LOCAL_EF"
echo "  M: $M"
echo "  Graph num: $GRAPH_NUM"
echo "  Threads: $THREADS"
echo ""

echo "Note: The improved implementation will:"
echo "  1. Build approximate εNG using Self_search (ef_for_εNG=100 by default)"
echo "  2. Construct MST on the εNG (more neighbors than original HNSW graph)"
echo "  3. Perform merge using MST-guided order"
echo ""

# Check if data and graphs exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Warning: Data file $DATA_FILE not found."
    echo "Please adjust DATA_FILE in this script to point to your dataset."
    echo ""
fi

if [ ! -f "${GRAPH_INDEX_FILE}1_ef${SUB_EF}_M${SUB_M}.hnsw" ]; then
    echo "Warning: Graph files not found at $GRAPH_INDEX_FILE"
    echo "Please adjust GRAPH_INDEX_FILE or build subgraphs first."
    echo ""
fi

echo "To run the test with default εNG parameters (ef_for_εNG=100):"
echo "./test_SIM_merge $DATA_FILE $GLOBAL_EF $LOCAL_EF $M $SUB_EF $SUB_M $GRAPH_NUM $GRAPH_INDEX_FILE $MERGED_NSG_PATH $MERGE_ORDER $THREADS"
echo ""

echo "To customize εNG construction, modify the code to set parameters:"
echo "  params.Set<unsigned>(\"ef_for_εNG\", 150);           // Get more neighbors"
echo "  params.Set<bool>(\"use_epsilon_filter\", true);      // Enable filtering"
echo "  params.Set<float>(\"epsilon_threshold\", 0.5);       // Distance threshold"
echo ""

echo "Expected output:"
echo "  - 'Building approximate εNG with ef=100'"
echo "  - 'MST construction completed with X edges'"
echo "  - 'SIM merge completed: 100%'"
echo "  - 'L : G = ...' (ratio of local to global searches)"
