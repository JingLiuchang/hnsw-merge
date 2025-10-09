#!/bin/bash
source params.sh
{
for partition_method in "${partition_methods[@]}"; do
  for db in "${datasets[@]}"; do
    if [ "$db" == "sift" ]; then
      ef=100
      M=32
      sub_ef=50
      sub_M=16
    elif [ "$db" == "deep1M" ]; then
      ef=80
      M=32
      sub_ef=40
      sub_M=16
    elif [ "$db" == "deep10M" ]; then
      ef=80
      M=32
      sub_ef=40
      sub_M=16
    elif [ "$db" == "gist" ]; then
      ef=120
      M=32
      sub_ef=60
      sub_M=16
    else
      echo "Unknown dataset: $db"
      exit 1
    fi

    for m in "${ms[@]}"; do
      # build sub-datasets
      if [ "$partition_method" == "kmeans" ]; then
        python /home/jlc/hnswlib/py/kmeans_partition.py --num_clusters $m --db $db
      elif [ "$partition_method" == "random" ]; then
        python /home/jlc/hnswlib/py/random_partition.py --num_clusters $m --db $db
      else
        echo "Unknown partition method: $partition_method. Use 'kmeans' or 'random'."
        exit 1
      fi

      # Build sub-indexes
      if [ "$m" -eq 2 ]; then
        mkdir -p "../data/${db}/${partition_method}/performance/bi"
        # sub indexes
        for part in $(seq 1 $m); do
          echo "part: $part"
          ../cmake-build-debug/test_hnsw_index \
          ../data/${db}/${partition_method}/bi-index-data/${db}_${partition_method}P${part}_base.fvecs \
          $sub_ef \
          $sub_M \
          ../data/${db}/${partition_method}/bi-index-merged/${db}_${partition_method}P${part}_ef${sub_ef}_M${sub_M}.hnsw
        done

        # BuildAsOne index
        ../cmake-build-debug/test_hnsw_level0_index \
        ../data/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_base.fvecs \
        $ef \
        $M \
        ../data/${db}/${partition_method}/bi-index-merged/${db}_${partition_method}_BuildAsOne_ef${ef}_M${M}.hnsw

        # compute groundtruth
        python /home/jlc/hnswlib/py/gt_gpu.py \
        --base_file ../data/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_base.fvecs \
        --query_file ../data/${db}/${db}_query.fvecs \
        --gt_file ../data/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_groundtruth.ivecs \
        --dist_file ../data/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_distance.fvecs

      elif [ "$m" -gt 2 ]; then

        mkdir -p "../data/${db}/${partition_method}/performance/${m}parts"
        # sub indexes
        for part in $(seq 1 $m); do
          echo "part: $part"
          ../cmake-build-debug/test_hnsw_index \
          ../data/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}P${part}_base.fvecs \
          $sub_ef \
          $sub_M \
          ../data/${db}/${partition_method}/multi-index-merged/${m}parts/${db}_${partition_method}P${part}_ef${sub_ef}_M${sub_M}.hnsw
        done

        # BuildAsOne index
        ../cmake-build-debug/test_hnsw_level0_index \
        ../data/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_base.fvecs \
        $ef \
        $M \
        ../data/${db}/${partition_method}/multi-index-merged/${m}parts/${db}_${partition_method}_BuildAsOne_ef${ef}_M${M}.hnsw

        # compute groundtruth
        python /home/jlc/hnswlib/py/gt_gpu.py \
        --base_file ../data/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_base.fvecs \
        --query_file ../data/${db}/${db}_query.fvecs \
        --gt_file ../data/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_groundtruth.ivecs \
        --dist_file ../data/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_distance.fvecs
      fi
    done
  done
done
} 2>&1 | tee build-subgraph.log