#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/params.sh"

if [[ $# -gt 0 ]]; then
  run_datasets=("$@")
else
  run_datasets=("${datasets[@]}")
fi

for T in "${Ts[@]}"; do
  for db in "${run_datasets[@]}"; do
    case "$db" in
      sift)
        ef=40; M=25; sub_ef=200; sub_M=25; PARAMS=("40 20 3") ;;
      deep1M)
        ef=40; M=30; sub_ef=200; sub_M=30; PARAMS=("40 10 3") ;;
      msong)
        ef=30; M=40; sub_ef=200; sub_M=40; PARAMS=("30 10 3") ;;
      msmarco1M)
        ef=50; M=30; sub_ef=200; sub_M=30; PARAMS=("50 10 3") ;;
      anton1m)
        ef=40; M=30; sub_ef=200; sub_M=30; PARAMS=("40 20 3") ;;
      imagenet1m)
        ef=40; M=30; sub_ef=200; sub_M=30; PARAMS=("40 10 3") ;;
      gist)
        ef=40; M=30; sub_ef=200; sub_M=30; PARAMS=("40 10 3") ;;
      deep10m)
        ef=40; M=30; sub_ef=300; sub_M=30; PARAMS=("40 5 3") ;;
      msmarc10m)
        ef=50; M=30; sub_ef=300; sub_M=30; PARAMS=("50 10 3") ;;
      imagenet10m)
        ef=50; M=30; sub_ef=300; sub_M=30; PARAMS=("50 10 3") ;;
      anton10m)
        ef=40; M=30; sub_ef=300; sub_M=30; PARAMS=("40 10 3") ;;
      *)
        echo "Unknown dataset: $db" >&2
        exit 1 ;;
    esac

    DATA_FILE="${DATA_PATH}/${db}/random/bi-index-data/${db}_random_base.fvecs"
    GRAPH_INDEX_FILE="${DATA_PATH}/${db}/random/bi-index-merged/${db}_randomP"
    INDEX_DIR="${DATA_PATH}/${db}/random/bi-index-merged"
    ET=0
    RATIO=1.0
    morder="pairwise"

    mkdir -p "${REPO_PATH}/logs/${db}"
    LOG_FILE="${REPO_PATH}/logs/${db}/RGTMmem_merge.log"
    echo "Starting graph-only index merge at $(date)" | tee -a "$LOG_FILE"

    for param in "${PARAMS[@]}"; do
      read -r G L S <<< "$param"
      RGTM_OUTPUT_FILE="${INDEX_DIR}/${db}_random_RGTMmem_et${ET}_ef${G}_${L}_${S}_M${M}_adjacent-blocks.hnsw"
      "${BUILD_DIR}/test_RGTMmem_merge" \
        "$DATA_FILE" "$G" "$L" "$S" "$M" "$sub_ef" "$sub_M" 2 \
        "$GRAPH_INDEX_FILE" "$RGTM_OUTPUT_FILE" "$ET" "$RATIO" "$morder" "$T" \
        2>&1 | tee -a "$LOG_FILE"
    done

    NGM_OUTPUT_FILE="${INDEX_DIR}/${db}_random_NGMmem_et${ET}_ef${ef}_M${M}.hnsw"
    "${BUILD_DIR}/test_NGMmem_merge" \
      "$DATA_FILE" "$ef" "$M" "$sub_ef" "$sub_M" 2 \
      "$GRAPH_INDEX_FILE" "$NGM_OUTPUT_FILE" "$ET" "$RATIO" "$morder" "$T" \
      2>&1 | tee -a "$LOG_FILE"

    echo "Graph-only index merge completed at $(date)" | tee -a "$LOG_FILE"
  done
done
