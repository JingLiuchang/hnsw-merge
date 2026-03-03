#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
    mkdir -p "/mnt/ssd/merge_bench/${db}"
    for method in "${partition_methods[@]}"; do
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/bi-index-data"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/bi-index-merged"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/logs"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/multi-index-data"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/multi-index-merged"
        mkdir -p "/mnt/ssd/merge_bench/${db}/${method}/performance"
    done
done