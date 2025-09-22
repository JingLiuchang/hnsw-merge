#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
    mkdir -p "../data/${db}"
    for method in "${partition_methods[@]}"; do
        mkdir -p "../data/${db}/${method}"
        mkdir -p "../data/${db}/${method}/bi-index-data"
        mkdir -p "../data/${db}/${method}/bi-index-merged"
        mkdir -p "../data/${db}/${method}/logs"
        mkdir -p "../data/${db}/${method}/multi-index-data"
        mkdir -p "../data/${db}/${method}/multi-index-merged"
        mkdir -p "../data/${db}/${method}/performance"
    done
done