<p align="center">
  <strong>CHSZLabLib</strong><br>
  <em>High-performance graph algorithms with a Pythonic interface</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/build-CMake%20%2B%20pybind11-brightgreen?logo=cmake" alt="CMake + pybind11">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License: MIT">
</p>

---

CHSZLabLib provides a unified Python interface to five established C/C++ graph algorithm libraries:

| Library | Capability | Reference |
|---------|-----------|-----------|
| [**KaHIP**](https://github.com/KaHIP/KaHIP) | Graph partitioning, node separators, nested dissection | Karlsruhe High Quality Partitioning |
| [**VieCut**](https://github.com/VieCut/VieCut) | Global minimum cuts | Vienna Minimum Cuts |
| [**VieClus**](https://github.com/VieClus/VieClus) | Graph clustering / community detection | Vienna Graph Clustering |
| [**CHILS**](https://github.com/KarlsruheMIS/CHILS) | Maximum weight independent set | Concurrent Heuristic Independent Local Search |
| [**KaMIS**](https://github.com/KarlsruheMIS/KaMIS) | Maximum independent set (weighted & unweighted) | Karlsruhe Maximum Independent Sets |

All algorithms operate on a shared `Graph` object backed by NumPy arrays in CSR format -- no data copying between tools.

## Quick Start

```python
from chszlablib import Graph, partition, mincut, cluster, mwis, redumis

# Build a graph
g = Graph(num_nodes=6)
for u, v in [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (2,3)]:
    g.add_edge(u, v)

# Partition into 2 blocks
p = partition(g, num_parts=2, mode="strong")
print(f"Edge cut: {p.edgecut}")
print(f"Assignment: {p.assignment}")

# Global minimum cut
mc = mincut(g, algorithm="viecut")
print(f"Min-cut value: {mc.cut_value}")

# Community detection
c = cluster(g, time_limit=1.0)
print(f"Modularity: {c.modularity:.4f}, clusters: {c.num_clusters}")

# Maximum weight independent set
for i in range(g.num_nodes):
    g.set_node_weight(i, i + 1)
m = mwis(g, time_limit=1.0)
print(f"MWIS weight: {m.weight}, vertices: {m.vertices}")

# Maximum independent set (unweighted)
r = redumis(g, time_limit=1.0)
print(f"MIS size: {r.size}, vertices: {r.vertices}")
```

## Installation

```bash
git clone --recursive https://github.com/CHSZLab/CHSZLabLib.git
cd CHSZLabLib
bash build.sh
```

**Requirements:** Python >= 3.9, a C/C++17 compiler (GCC or Clang), CMake >= 3.15, OpenMP (optional, for parallelism).

The build script creates a virtualenv in `.venv/`, compiles all native extensions, installs the wheel, and runs the test suite.

## Graph Construction

### From edge list

```python
g = Graph(num_nodes=4)
g.add_edge(0, 1)
g.add_edge(1, 2, weight=5)    # optional edge weight
g.set_node_weight(0, 10)       # optional node weight
g.finalize()                    # auto-called on first property access
```

### From CSR arrays

```python
import numpy as np
g = Graph.from_csr(
    xadj=np.array([0, 2, 4, 6, 8]),
    adjncy=np.array([1, 3, 0, 2, 1, 3, 0, 2]),
)
```

### From METIS file

```python
g = Graph.from_metis("my_graph.graph")
g.to_metis("output.graph")
```

## API Reference

### Graph Partitioning

```python
partition(g, num_parts=2, mode="eco", imbalance=0.03, seed=0) -> PartitionResult
```

Partition a graph into *k* balanced blocks using KaHIP.

| Parameter | Description |
|-----------|-------------|
| `num_parts` | Number of blocks |
| `mode` | `"fast"`, `"eco"`, `"strong"`, `"fastsocial"`, `"ecosocial"`, `"strongsocial"` |
| `imbalance` | Allowed imbalance (0.03 = 3%) |

Returns `PartitionResult` with `edgecut` (int) and `assignment` (ndarray of block IDs).

### Node Separator

```python
node_separator(g, num_parts=2, mode="eco", imbalance=0.03, seed=0) -> SeparatorResult
```

Find a small set of nodes whose removal disconnects the graph into *k* parts.

Returns `SeparatorResult` with `num_separator_vertices` (int) and `separator` (ndarray of node indices).

### Nested Dissection Ordering

```python
node_ordering(g, mode="eco", seed=0) -> OrderingResult
```

Compute a fill-reducing ordering for sparse matrix factorization.

Returns `OrderingResult` with `ordering` (ndarray permutation).

### Global Minimum Cut

```python
mincut(g, algorithm="viecut", seed=0) -> MincutResult
```

Compute the global minimum cut.

| Algorithm | Aliases |
|-----------|---------|
| VieCut | `"viecut"`, `"vc"` |
| NOI | `"noi"` |
| Karger-Stein | `"ks"` |
| Matula | `"matula"` |
| Padberg-Rinaldi | `"pr"` |
| Cactus | `"cactus"` |

Returns `MincutResult` with `cut_value` (int) and `partition` (0/1 ndarray).

### Graph Clustering

```python
cluster(g, time_limit=1.0, seed=0, cluster_upperbound=0) -> ClusterResult
```

Detect communities by maximizing modularity using VieClus.

| Parameter | Description |
|-----------|-------------|
| `time_limit` | Seconds to optimize (default 1.0) |
| `cluster_upperbound` | Max cluster size (0 = unlimited) |

Returns `ClusterResult` with `modularity` (float), `num_clusters` (int), and `assignment` (ndarray).

### Maximum Weight Independent Set

```python
mwis(g, time_limit=10.0, num_concurrent=4, seed=0) -> MWISResult
```

Find a maximum weight independent set using CHILS.

| Parameter | Description |
|-----------|-------------|
| `time_limit` | Seconds to optimize (default 10.0) |
| `num_concurrent` | Parallel solution attempts (default 4) |

Returns `MWISResult` with `weight` (int) and `vertices` (ndarray of node indices).

### Maximum Independent Set (KaMIS)

KaMIS provides four algorithms covering both unweighted and weighted MIS problems. All return `MISResult` with `size` (int), `weight` (int), and `vertices` (ndarray of node indices).

#### Unweighted MIS

```python
redumis(g, time_limit=10.0, seed=0, full_kernelization=False) -> MISResult
```

Evolutionary algorithm with graph reductions. Best solution quality for unweighted MIS.

| Parameter | Description |
|-----------|-------------|
| `time_limit` | Seconds to optimize (default 10.0) |
| `full_kernelization` | Use full kernelization (`True`) or FastKer (`False`, default) |

```python
online_mis(g, time_limit=10.0, seed=0, ils_iterations=15000) -> MISResult
```

Iterated local search. Faster than ReduMIS, useful for quick approximate solutions.

| Parameter | Description |
|-----------|-------------|
| `time_limit` | Seconds to optimize (default 10.0) |
| `ils_iterations` | Number of ILS iterations (default 15000) |

#### Weighted MIS

```python
branch_reduce(g, time_limit=10.0, seed=0) -> MISResult
```

Exact branch-and-reduce solver for maximum weight independent set. Uses node weights from the graph.

```python
mmwis_solver(g, time_limit=10.0, seed=0) -> MISResult
```

Memetic evolutionary algorithm for maximum weight independent set. Trades exactness for scalability on larger graphs.

## I/O

Read and write graphs in [METIS format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf):

```python
from chszlablib import read_metis, write_metis

g = read_metis("graph.metis")   # or Graph.from_metis(...)
write_metis(g, "output.metis")  # or g.to_metis(...)
```

Supports unweighted, edge-weighted, node-weighted, and fully weighted graphs.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Project Structure

```
CHSZLabLib/
  chszlablib/         Python package (graph, partition, mincut, cluster, mwis, mis, io)
  bindings/           pybind11 C++ binding code
  KaHIP/              KaHIP submodule
  VieCut/             VieCut submodule
  VieClus/            VieClus submodule
  CHILS/              CHILS submodule
  KaMIS/              KaMIS submodule (ReduMIS, OnlineMIS, Branch&Reduce, MMWIS)
  tests/              pytest suite (82 tests)
  CMakeLists.txt      Top-level build configuration
  build.sh            One-step build script
```
