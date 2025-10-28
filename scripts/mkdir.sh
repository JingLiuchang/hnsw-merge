#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
    mkdir -p "/mnt/ssd/merge_bench/${db}"
    mkdir -p "/home/jlc/hnsw-merge/performance/${db}"
    for method in "${partition_methods[@]}"; do
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/bi-index-data"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/bi-index-merged"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/logs"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/multi-index-data"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/multi-index-merged"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/performance"
        for m in "${ms[@]}"; do
          mkdir -p "/home/jlc/hnsw-merge/performance/${db}/${m}parts"
        done
    done
done