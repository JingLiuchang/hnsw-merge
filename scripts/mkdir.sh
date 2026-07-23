#!/bin/bash
source params.sh

for db in "${datasets[@]}"; do
    mkdir -p "${DATA_PATH}/${db}"
    for method in "${partition_methods[@]}"; do
        mkdir -p "${DATA_PATH}/${db}/${method}"
        mkdir -p "${DATA_PATH}/${db}/${method}/bi-index-data"
        mkdir -p "${DATA_PATH}/${db}/${method}/bi-index-merged"
        mkdir -p "${DATA_PATH}/${db}/${method}/logs"
        mkdir -p "${DATA_PATH}/${db}/${method}/multi-index-data"
        mkdir -p "${DATA_PATH}/${db}/${method}/multi-index-merged"
        mkdir -p "${DATA_PATH}/${db}/${method}/performance"
    done
done