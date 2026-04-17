# Reverse Neighbor Sliding and Order Selection for Efficient Multi-Proximity Graph Merging

## Introduction

This is the official implementation of the paper [Reverse Neighbor Sliding and Order Selection for Efficient Multi-Proximity Graph Merging].

RNSM+ supports efficient two-index merging and the merge order of multiple indexes and leverage these techniques for scalable index construction. RNSM+ yields up to a 2.65$\times$ indexing speedup over existing methods, while maintaining expected superior search performance. Moreover, our method scales efficiently to 100 million vectors with 50 partitions, maintaining consistent speedups.

## Requirements

* C++17
* Python
* OpenMP

## Directory Structure

```
.
├── examples
│   └── cpp                                  # Code for running our method
├── hnswlib                                        # HNSW library
├── py                                             # Python function for dataprocessing, partitioning, and MOS
└── scripts                                        # Scripts for reproduction
```

## Reproduction

### Prepare Datasets

All datasets we used for the evaluation can be downloaded from [ANN-Benchmark](https://github.com/erikbern/ann-benchmarks) or other public repo. Download the base and query set to the your directory and ensure they are in `.fvecs`/`.ivecs` format.

### Runing Tests

Step 1. Specify datasets, partition number, and partition method in `scripts/params.sh`

```zsh
REPO_PATH="path_to_your_repo"
DATA_PATH="path_to_your_datasets"
datasets=("deep10m" "msmarc10m" "anton10m" "imagenet10m")
partition_methods=("random") # random / kmeans / overlapping
ms=(8) # number of partitions
```
Step 2. Construct sub-indexes

To run tests on new datasets, you must construct sub-indexes first. You can do this by running `scripts/beforeMerge.sh`, which will partition the dataset and build sub-indexes for each partition.

Indexing parameter specimens are listed below.
- `sub_ef`: ef used for constructing sub-indexes on each partition.
- `sub_M`: maximum out-degree of each sub-index.

Step 3. Choose which test you wish to run.

Tests include:

* `scripts/test_RGTM_merge.sh`: merging random partitions.
* `scripts/test_RGTM_skewed_merge.sh`: merging skewed partitions.

*"Note: RGTM / NGM is the internal code name for the RNSM / CM described in the paper."*
  

