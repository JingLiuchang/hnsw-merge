# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research codebase extending hnswlib with graph merging algorithms for HNSW (Hierarchical Navigable Small World) indices. The project implements multiple merge strategies for combining partitioned HNSW graphs. Currently on branch `parallelism-exp`, comparing NSM (sequential/chain-dependent) vs SIM/RNSM (parallelism-friendly) merge algorithms.

## Build

```bash
mkdir build && cd build && cmake .. && make -j$(nproc)
```

Requires TBB (Threading Building Blocks). Build produces executables in `build/`:
- `test_RGTM_merge`, `test_NGM_merge`, `test_NSM_merge`, `test_SIM_merge`, `test_OVERLAP_merge`
- `test_hnsw_index`, `test_hnsw_search`, `test_hnsw_level0_index`

## Running Experiments

Shell scripts in `scripts/` source `scripts/params.sh` for dataset/parameter configuration:

```bash
cd scripts
bash test_RGTM_bimerge.sh    # RGTM pairwise merge
bash test_NSM_bimerge.sh     # NSM merge
bash test_SIM_bisearch.sh    # SIM (RNSM) merge
```

Executables take positional CLI args. Example for `test_NSM_merge`:
```
data_file global_ef local_ef M sub_ef sub_M graph_num graph_index_file merged_nsg_path k_plus merge_order_selection [T] [merge_order_file]
```

For `test_RGTM_merge`:
```
data_file global_ef local_ef self_ef M sub_ef sub_M graph_num graph_index_file merged_nsg_path ET ratio merge_order_selection [T] [merge_order_file]
```

Subgraph index files are named with pattern: `{prefix}{i}_ef{sub_ef}_M{sub_M}.hnsw`

## Architecture

### Core Headers (`hnswlib/`)

**`hnswalg.h`**: Base `HierarchicalNSW<dist_t>` class. Manages graph structure via `data_level0_memory_` (level-0 links + data) and `linkLists_` (higher-level links). Uses `label_lookup_` (external label → internal `tableint` ID).

**`mergealg.h`**: `MergeHierarchicalNSW<dist_t>` extending `HierarchicalNSW`. The central dispatch is `mgraph_merge(m, graphs, parameters)` which reads a `"method"` parameter string and calls the appropriate implementation. Also adds:
- `mergeid_lookup_`: merge ID → `(local_id, graph_id)` pairs
- `globalid_offset`: cumulative element counts across subgraphs

**`parameter.h`**: `Parameters` key-value store, `reverseNN_info`, `block_info` structs.

**`utils.h`**: Data loading (`load_data`, `safe_load_data` for fvecs/bvecs format).

### Merge Algorithms (all in `mergealg.h`)

All algorithms are dispatched from `mgraph_merge()` via the `"method"` parameter:

| Method key | Function | Description |
|---|---|---|
| `"NGM"` | `NGM_merge_into_later` | Naive graph merge — insert G1 nodes into G2 via full search |
| `"RGTM"` | `RGTM_merge_into_later` | Reverse Graph Traversal Merge — uses reverse NN traversal |
| `"NSM"` | `NSM_merge_into_later` | Neighbor Sliding Merge — chain-dependent sliding (sequential) |
| `"NSM_OPT"` | `NSM_merge_into_later_optimized` | Optimized NSM variant |
| `"SIM"` | `SIM_merge_into_later` | Set-cover greedy pivot selection enabling full parallelism (RNSM) |
| `"SIM_OPT"` | `SIM_merge_into_later_optimized` | Optimized SIM variant |

**Key algorithmic distinction (parallelism-exp branch)**: SIM/RNSM pre-selects pivots via greedy set-cover on RNN graph before merging, allowing fully parallel execution. NSM creates chain dependencies (each follower depends on previous result), limiting parallelism.

### Merge Order Selection

`mgraph_merge` also reads `"merge_order_selection"` to determine pairwise merge sequence:
- `"pairwise"`: sequential pairs (0,1), (1,2), …
- `"circle"`: circular/ring order
- `"mst"`: order from MST file (requires `"merge_order_file"` path)

### Test Programs Pattern (`examples/cpp/`)

Each `test_*_merge.cpp`:
1. Parses CLI args
2. `load_data()` or `safe_load_data()` the full dataset
3. Loads subgraph `.hnsw` files into a `std::vector<HierarchicalNSW<float>*>`
4. Creates `MergeHierarchicalNSW<float>` and calls `mgraph_merge()`
5. Saves merged index and runs recall evaluation

## Key Files for Current Work

- `PARALLELISM_EXP_SETUP.md`: Algorithm specs (NSM and SIM/RNSM pseudocode) and tasks for `parallelism-exp` branch
- `scripts/test_NSM_bimerge.sh`, `scripts/test_SIM_bisearch.sh`: Experiment scripts for parallelism comparison
- `hnswlib/mergealg.h`: All merge algorithm implementations (3000+ lines)

## Dataset Configuration

Datasets configured in `scripts/params.sh`. Common parameters:

| Dataset | Size | Dim | M | ef |
|---|---|---|---|---|
| sift | 1M | 128 | 25 | 40 |
| deep1M | 1M | 96 | 30 | 40 |
| deep10m | 10M | 96 | 30 | 300 |
| msong | 990K | 420 | 40 | 30 |
| msmarco1M | 1M | 1024 | 30 | 50 |
| gist | 1M | 960 | 30 | 40 |

Data files are in fvecs/bvecs format. Performance logs go to `performance/` (gitignored).
