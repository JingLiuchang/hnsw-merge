#!/bin/bash

#bash test_RGTM_bimerge.sh
#bash test_RGTM_bisearch.sh
#bash /home/jlc/pg-fast-merging/scripts/run.sh

#bash beforeMerge.sh
#bash test_RGTM_bimerge.sh


datasets=("anton1m") #"sift" "deep1M" "gist" "msmarco1M" "anton1m" "imagenet1m"
for db in "${datasets[@]}"; do
  if [ "$db" == "sift" ]; then
    ef=200
    M=25
    sub_ef=200
    sub_M=25
  elif [ "$db" == "deep1M" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  elif [ "$db" == "msong" ]; then
    ef=200
    M=40
    sub_ef=200
    sub_M=40
  elif [ "$db" == "msmarco1M" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  elif [ "$db" == "anton1m" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  elif [ "$db" == "imagenet1m" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  elif [ "$db" == "gist" ]; then
    ef=200
    M=30
    sub_ef=200
    sub_M=30
  else
    echo "Unknown dataset: $db"
    exit 1
  fi

  # /home/jlc/hnsw-merge/cmake-build-debug/test_hnsw_level0_index /mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_base.fvecs $ef $M /mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_random_BuildAsOne_ef${ef}_M${M}.hnsw 2>&1 | tee -a /home/jlc/pg-fast-merging/performance/${db}/build-subgraph.log
  /home/jlc/hnsw-merge/cmake-build-debug/test_hnsw_search /mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_base.fvecs /mnt/ssd/merge_bench/${db}/${db}_query.fvecs /mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_groundtruth.ivecs /mnt/ssd/merge_bench/${db}/random/bi-index-merged/${db}_random_BuildAsOne_ef${ef}_M${M}.hnsw 10 10 150 10 /home/jlc/pg-fast-merging/performance/${db}/${db}_random_BuildAsOne_ef${ef}_M${M}.csv
done