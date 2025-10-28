#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
  for partition_method in "${partition_methods[@]}"; do
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
    elif [ "$db" == "anton10m" ]; then
      ef=300
      M=30
      sub_ef=300
      sub_M=30
    elif [ "$db" == "imagenet10m" ]; then
      ef=300
      M=30
      sub_ef=300
      sub_M=30
    elif [ "$db" == "deep10m" ]; then
      ef=300
      M=30
      sub_ef=300
      sub_M=30
    elif [ "$db" == "msmarc10m" ]; then
      ef=300
      M=30
      sub_ef=300
      sub_M=30
    elif [ "$db" == "deep100M" ]; then
          ef=500
          M=30
          sub_ef=500
          sub_M=30
    else
      echo "Unknown dataset: $db"
      exit 1
    fi

    for m in "${ms[@]}"; do
      mkdir -p "/home/jlc/hnsw-merge/performance/${db}/${m}parts/"
      {
      # build sub-merge_bench
      if [ "$partition_method" == "kmeans" ]; then
        python /home/jlc/hnsw-merge/py/kmeans_partition.py --num_clusters $m --db $db
      elif [ "$partition_method" == "random" ]; then
        python /home/jlc/hnsw-merge/py/random_partition.py --num_clusters $m --db $db
      else
        echo "Unknown partition method: $partition_method. Use 'kmeans' or 'random'."
        exit 1
      fi

      # Build sub-indexes
      if [ "$m" -eq 2 ]; then
        mkdir -p "/mnt/ssd/merge_bench/${db}/${partition_method}/performance/bi"
        # sub indexes
        for part in $(seq 1 $m); do
          echo "part: $part"
          ../cmake-build-debug/test_hnsw_index \
          /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-data/${db}_${partition_method}P${part}_base.fvecs \
          $sub_ef \
          $sub_M \
          /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-merged/${db}_${partition_method}P${part}_ef${sub_ef}_M${sub_M}.hnsw
        done

        # BuildAsOne index
        ../cmake-build-debug/test_hnsw_level0_index \
        /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_base.fvecs \
        $ef \
        $M \
        /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-merged/${db}_${partition_method}_BuildAsOne_ef${ef}_M${M}.hnsw

        # compute groundtruth
#        python /home/jlc/hnsw-merge/py/gt_gpu.py \
#        --base_file /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_base.fvecs \
#        --query_file /mnt/ssd/merge_bench/${db}/${db}_query.fvecs \
#        --gt_file /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_groundtruth.ivecs \
#        --dist_file /mnt/ssd/merge_bench/${db}/${partition_method}/bi-index-data/${db}_${partition_method}_distance.fvecs

      elif [ "$m" -gt 2 ]; then

        mkdir -p "/mnt/ssd/merge_bench/${db}/${partition_method}/performance/${m}parts"
        # sub indexes
        for part in $(seq 1 $m); do
          echo "part: $part"
          ../cmake-build-debug/test_hnsw_index \
          /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}P${part}_base.fvecs \
          $sub_ef \
          $sub_M \
          /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-merged/${m}parts/${db}_${partition_method}P${part}_ef${sub_ef}_M${sub_M}.hnsw
        done

        # BuildAsOne index
        ../cmake-build-debug/test_hnsw_level0_index \
        /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_base.fvecs \
        $ef \
        $M \
        /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-merged/${m}parts/${db}_${partition_method}_BuildAsOne_ef${ef}_M${M}.hnsw

#        # compute groundtruth
#        python /home/jlc/hnsw-merge/py/gt_gpu.py \
#        --base_file /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_base.fvecs \
#        --query_file /mnt/ssd/merge_bench/${db}/${db}_query.fvecs \
#        --gt_file /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_groundtruth.ivecs \
#        --dist_file /mnt/ssd/merge_bench/${db}/${partition_method}/multi-index-data/${m}parts/${db}_${partition_method}_distance.fvecs
      fi
      } 2>&1 | tee -a /home/jlc/hnsw-merge/performance/${db}/${m}parts/build-subgraph.log
    done
  done
done