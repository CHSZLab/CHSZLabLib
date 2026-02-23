# chszlablib: Unified Python Interface Design

**Date:** 2026-02-23
**Status:** Approved

## Problem

Four algorithm libraries (KaHIP, VieCut, VieClus, CHILS) each provide graph algorithms with different APIs, build systems, and data structures. We need a single Python package exposing all algorithms through a common graph representation.

## Constraints

- Libraries themselves must NOT be modified; all glue code lives outside their directories
- Linux, macOS, and Windows support required
- Fresh pybind11 bindings for all four (not wrapping existing pip packages)
- Single pip-installable package called `chszlablib`
- Custom lightweight Graph class backed by CSR arrays

## Libraries Overview

| Library | Problem Domain | Language | Build | Internal Graph |
|---------|---------------|----------|-------|----------------|
| KaHIP | Graph partitioning (6 modes), node separators, node ordering, edge partitioning, process mapping | C++ | CMake | CSR (xadj/adjncy) |
| VieCut | Global minimum cut (7 algorithms), multiterminal cut, dynamic min-cut | C++17 | CMake | CSR (graph_access + mutable_graph) |
| VieClus | Graph clustering (Louvain, label propagation, memetic) | C++11 | CMake | CSR (xadj/adjncy, reuses KaHIP) |
| CHILS | Maximum weight independent set (MWIS) | C17 | Makefile | CSR (V/E/W arrays) |

All use CSR-style representations and METIS file format.

## Architecture

**Approach:** External pybind11 modules linking against library builds. Each library is built as a static library via its own build system. pybind11 modules link against these. A pure-Python layer provides the unified Graph class and algorithm wrappers.

## Repository Layout

```
CHSZLabLib/
├── KaHIP/              # untouched
├── VieCut/             # untouched
├── VieClus/            # untouched
├── CHILS/              # untouched
├── bindings/           # pybind11 glue code
│   ├── kahip_bind.cpp      # wraps kaHIP_interface.h functions
│   ├── viecut_bind.cpp     # wraps VieCut algorithm classes
│   ├── vieclus_bind.cpp    # wraps vieclus_interface.h functions
│   └── chils_bind.cpp      # wraps chils.h functions
├── chszlablib/         # pure Python package
│   ├── __init__.py     # public API exports
│   ├── graph.py        # Graph class
│   ├── partition.py    # KaHIP wrappers
│   ├── mincut.py       # VieCut wrappers
│   ├── cluster.py      # VieClus wrappers
│   ├── mwis.py         # CHILS wrappers
│   └── io.py           # METIS file I/O
├── CMakeLists.txt      # top-level build orchestration
├── pyproject.toml      # scikit-build-core config
└── tests/
    ├── test_graph.py
    ├── test_partition.py
    ├── test_mincut.py
    ├── test_cluster.py
    └── test_mwis.py
```

## Graph Class (`chszlablib/graph.py`)

Custom lightweight class backed by numpy arrays in CSR format.

### Construction

```python
from chszlablib import Graph

# Builder API
g = Graph(num_nodes=5, directed=False)
g.add_edge(0, 1, weight=3)
g.add_edge(1, 2, weight=5)
g.set_node_weight(0, 10)

# From CSR arrays
g = Graph.from_csr(xadj, adjncy, node_weights=vwgt, edge_weights=adjcwgt)

# From METIS file
g = Graph.from_metis("graph.metis")
```

### Internals

- Stores CSR arrays: `xadj` (int64), `adjncy` (int32), `node_weights` (int64), `edge_weights` (int64)
- Builder mode accumulates edges in a list; finalizes to CSR on first algorithm call or explicit `finalize()`
- Undirected graphs store each edge twice (both directions) in CSR
- Default weights: nodes=1, edges=1

### Interop

```python
g.to_networkx()           # -> networkx.Graph (optional dependency)
g.to_scipy_sparse()       # -> scipy.sparse.csr_matrix
Graph.from_networkx(nx_g)
g.to_metis("out.graph")   # write METIS format
```

### Properties

```python
g.num_nodes       # int
g.num_edges       # int (undirected count)
g.xadj            # numpy array
g.adjncy          # numpy array
g.node_weights    # numpy array
g.edge_weights    # numpy array
```

## Algorithm APIs

### Partitioning (KaHIP)

```python
from chszlablib import partition

result = partition(g, num_parts=4, mode="eco", imbalance=0.03, seed=42)
result.edgecut       # int
result.assignment    # numpy array of partition IDs per node
```

**Modes:** `"fast"`, `"eco"`, `"strong"`, `"fastsocial"`, `"ecosocial"`, `"strongsocial"`

Additional functions:
- `node_separator(g, num_parts, mode, imbalance, seed)` -> `.num_separator_vertices`, `.separator`
- `node_ordering(g, mode, seed)` -> `.ordering`
- `edge_partition(g, num_parts, mode, imbalance, seed)` -> `.vertexcut`, `.assignment`

### Minimum Cut (VieCut)

```python
from chszlablib import mincut

result = mincut(g, algorithm="viecut")
result.cut_value    # int (weight of minimum cut)
result.partition    # numpy array (0 or 1 per node)
```

**Algorithms:** `"viecut"`, `"noi"`, `"ks"`, `"matula"`, `"pr"`, `"cactus"`

### Clustering (VieClus)

```python
from chszlablib import cluster

result = cluster(g, time_limit=5.0, seed=0)
result.modularity    # float
result.assignment    # numpy array of cluster IDs per node
```

### Maximum Weight Independent Set (CHILS)

```python
from chszlablib import mwis

result = mwis(g, time_limit=10.0, num_concurrent=4, seed=0)
result.weight      # int (total weight of independent set)
result.vertices    # numpy array of vertex IDs in the set
```

## Build System

### Top-level CMakeLists.txt

1. `add_subdirectory(KaHIP)` - builds `libkaffpa` static library
2. `add_subdirectory(VieCut)` - builds VieCut static library
3. `add_subdirectory(VieClus)` - builds VieClus static library
4. Custom target for CHILS: compiles its 5 .c source files (`graph.c`, `chils.c`, `chils_internal.c`, `local_search.c`) into a static lib (CHILS uses Makefile, but sources are simple enough for a CMake target)
5. `FetchContent(pybind11)` - fetches pybind11
6. Builds 4 pybind11 modules: `_kahip`, `_viecut`, `_vieclus`, `_chils`
7. Each module links against its respective static library

### pyproject.toml

Uses `scikit-build-core` as build backend:

```toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "chszlablib"
dependencies = ["numpy"]
optional-dependencies.networkx = ["networkx"]
optional-dependencies.scipy = ["scipy"]
```

### CMake Integration Details

- KaHIP: Already exports `libkaffpa` target. Link against it + include `interface/`
- VieClus: Has `vieclus_interface.h`. Build static lib from its sources. Include KaHIP extern headers.
- VieCut: No C interface exists. The pybind11 module will directly instantiate C++ algorithm classes (e.g., `minimum_cut`, `viecut`) and call them. Include paths point into `lib/`.
- CHILS: Pure C with clean `chils.h` API. Compile sources, link, expose via pybind11 `extern "C"` wrapping.

## Cross-Platform Support

### Linux
- GCC or Clang, full OpenMP support
- Primary development platform

### macOS
- Clang with OpenMP via `brew install libomp`
- CMake finds OpenMP via `find_package(OpenMP)`
- Fallback: disable OpenMP (sequential mode)

### Windows
- MSVC or MinGW
- OpenMP supported by MSVC natively
- May need adjustments for C99 features in CHILS (MSVC C support is limited)

### CI/CD
- GitHub Actions matrix: Linux (ubuntu), macOS (arm64), Windows
- `cibuildwheel` for building wheels across platforms
- Test matrix: Python 3.9-3.13

### MPI
- MPI-dependent features (KaFFPaE, ParHIP, distributed edge partitioning) disabled by default
- Only sequential/shared-memory algorithms exposed

## Error Handling

- Invalid graph state (e.g., calling algorithm before adding edges): Python `ValueError`
- Algorithm failures: Python `RuntimeError` with descriptive message
- Out-of-memory: let C++ exceptions propagate (pybind11 translates to Python exceptions)

## Testing Strategy

- Unit tests per module (graph, partition, mincut, cluster, mwis)
- Test with small known graphs (e.g., complete graphs, paths, cycles)
- Verify algorithm results against known optimal values
- Cross-platform CI
