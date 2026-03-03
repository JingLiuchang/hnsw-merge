# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase extending the hnswlib library with graph merging algorithms for HNSW (Hierarchical Navigable Small World) indices. The project implements multiple merge strategies (NGM, RGTM, OVERLAP) for combining partitioned HNSW graphs.

## Build System

Build the project using CMake:

```bash
mkdir build
cd build
cmake ..
make
```

Key executables built:
- `test_RGTM_merge` - RGTM (Reverse Graph Traversal Merge) algorithm
- `test_NGM_merge` - NGM (Naive Graph Merge) algorithm
- `test_OVERLAP_merge` - Overlapping partition merge
- `test_hnsw_index` - Basic HNSW index construction
- `test_hnsw_search` - HNSW search testing
- Standard hnswlib examples (example_search, example_filter, etc.)

The build requires TBB (Threading Building Blocks) library for parallel operations.

## Architecture

### Core Components

**hnswlib/hnswalg.h**: Base `HierarchicalNSW` class implementing standard HNSW algorithm
- Manages graph structure with `data_level0_memory_` and `linkLists_`
- Uses `label_lookup_` to map external labels to internal tableint IDs
- Thread-safe operations with `label_op_locks_` and `link_list_locks_`

**hnswlib/mergealg.h**: `MergeHierarchicalNSW` class extending HierarchicalNSW
- Adds merge-specific data structures:
  - `mergeid_lookup_`: Maps merge IDs to (local ID, graph ID) pairs
  - `globalid_offset`: Tracks element offsets across merged graphs
- Implements three merge strategies: NGM, RGTM, and OVERLAP

**hnswlib/parameter.h**: Parameter management and data structures
- `reverseNN_info`: Stores reverse nearest neighbor information for merge operations
- `block_info`: Manages partition block membership
- `Parameters`: Generic key-value parameter storage

**hnswlib/utils.h**: Utility functions for data loading and I/O operations

### Merge Algorithms

The codebase implements three graph merging approaches:

1. **NGM (Naive Graph Merge)**: Simple concatenation of subgraphs
2. **RGTM (Reverse Graph Traversal Merge)**: Uses reverse NN traversal with configurable parameters (global_ef, local_ef, self_ef)
3. **OVERLAP**: Handles overlapping partitions with special boundary handling

### Test Programs

Test programs in `examples/cpp/` follow this pattern:
- Load dataset using `load_data()` from utils.h
- Accept command-line parameters for ef_construction, M, merge parameters
- Build or load subgraph indices
- Execute merge algorithm
- Output merged index to file

Example: `test_RGTM_merge` takes 14-16 arguments including data file, ef parameters, M values, graph paths, merge order selection, and optional threading/order file parameters.

## Running Tests

### Shell Scripts

Scripts in `scripts/` automate testing workflows:

**scripts/params.sh**: Central configuration for datasets and parameters
- Defines dataset arrays, partition methods, and parameter ranges
- Sources conda environment setup

**scripts/build-subgraph.sh**: Builds subgraph indices for datasets
- Iterates over datasets and partition methods
- Sets dataset-specific ef and M parameters
- Calls index construction executables

**scripts/test_RGTM_merge.sh**: Runs RGTM merge experiments
- Commented out but shows parameter sweep patterns
- Logs results to performance directories

### Python Utilities

Python scripts in `py/` handle data preparation and analysis:
- `random_partition.py`: Random dataset partitioning
- `kmeans_partition.py`, `kmeans_overlapping_partition.py`: K-means based partitioning
- `merge_time_extract.py`: Extract timing data from logs
- `plot-performance.py`: Visualize performance results
- `utils.py`: Common utility functions

## Dataset Configuration

Supported datasets (from params.sh and build-subgraph.sh):
- sift: (1M, 128-dim), M=25
- deep1M: (1M, 96-dim), M=30
- deep10m: (10M, 96-dim), M=30, ef=300
- msong: (990K, 420-dim), M=40
- msmarco1M: (1M, 1024-dim), M=30
- gist: (1M, 960-dim), M=30
- anton1m, imagenet1m: M=30

Each dataset has specific ef_construction and M parameters tuned for performance.

## Key Concepts

**Merge ID System**: The merge algorithms use a two-level ID mapping:
- External labels (labeltype) → Internal IDs (tableint) via `label_lookup_`
- Merge IDs (mergeidtype) → (local ID, graph ID) pairs via `mergeid_lookup_`

**Partition Methods**:
- Random partitioning
- K-means clustering (with/without overlap)
- Configurable number of partitions (T parameter)

**Merge Order Selection**: RGTM supports different merge order strategies:
- "pairwise": Sequential pairwise merging
- Custom order from file

## Development Notes

- This is a header-only C++ library (hnswlib core)
- Test programs are compiled separately with TBB linkage
- Performance logs stored in `performance/` directory (gitignored)
- Build artifacts in `cmake-build-debug/` and `build/` (gitignored)
