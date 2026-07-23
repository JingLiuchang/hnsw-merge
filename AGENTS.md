# Repository Guidelines

## Project Structure & Module Organization

This repository is a header-only C++ HNSW library with Python bindings. Core headers are in `hnswlib/`; Python binding sources and tests are in `python_bindings/` and `tests/python/`. Standalone C++ examples, merge implementations, and benchmark entry points are in `examples/cpp/`. C++ unit tests live in `tests/cpp/`. Dataset preparation and experiment runners are under `py/` and `scripts/`; generated benchmark logs and figures belong under `results/` and should not be mixed with source changes.

## Build, Test, and Development Commands

- `cmake -S . -B build -DHNSWLIB_EXAMPLES=ON && cmake --build build -j`: configure and compile examples, merge tools, and C++ tests.
- `make test`: run the Python binding unit-test suite from `tests/python/`.
- `python -m unittest discover -v --start-directory examples/python --pattern 'example*.py'`: exercise Python examples.
- `scripts/build-subgraph.sh`: create configured partitions and indexes; review `scripts/params.sh` and dataset paths before running because it can write large external artifacts.
- `git diff --check`: catch whitespace errors before submitting changes.

## Coding Style & Naming Conventions

Use four spaces in Python and two or four spaces consistently with the surrounding C++ file. Keep C++11-compatible code, braces, and naming consistent with nearby `hnswlib` code. Use `snake_case` for Python functions and variables, `PascalCase` for C++ types, and descriptive lowercase script names. Preserve existing shell quoting and `set -euo pipefail` patterns. Run the relevant formatter or linter already used by the touched component; do not introduce a new tool for a narrow change.

## Testing Guidelines

Add or update a focused test when changing behavior. Python tests use `unittest` and files named `bindings_test*.py`; C++ tests are executable targets under `tests/cpp/`. For merge or dataset experiments, record the exact dataset, parameters, executable, and output path, and validate generated CSV/log files rather than committing large indexes.

## Commit & Pull Request Guidelines

Recent history uses short imperative/topic summaries such as `NDC` and `NGM bi-merge implemented`; keep commits concise and scoped to one change. A pull request should explain the behavior change, list commands and tests run, link the relevant issue or experiment note, and include figures or screenshots when output changes. Call out dataset paths, thread counts, and known measurement limits for benchmark work. Do not overwrite user-generated results or unrelated worktree changes.
