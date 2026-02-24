<h1 align="center">CHSZLabLib</h1>

<p align="center">
  <strong>A unified Python interface to high-performance graph algorithms</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%E2%89%A5%203.9-3776ab?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/C%2B%2B17-pybind11-00599C?logo=cplusplus&logoColor=white" alt="C++17 / pybind11">
  <img src="https://img.shields.io/badge/build-CMake%20%2B%20scikit--build-064F8C?logo=cmake&logoColor=white" alt="CMake + scikit-build">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

<p align="center">
  <em>
    11 state-of-the-art graph algorithm libraries.&nbsp;
    One <code>Graph</code> object.&nbsp;
    Zero-copy NumPy arrays.
  </em>
</p>

---

## Table of Contents

- [Why CHSZLabLib?](#why-chszlablib)
- [Integrated Libraries](#integrated-libraries)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Graph Construction](#graph-construction)
- [Algorithms & API Reference](#algorithms--api-reference)
  - [Graph Partitioning (KaHIP)](#graph-partitioning-kahip)
  - [Evolutionary Partitioning (KaHIP)](#evolutionary-partitioning-kahip)
  - [Global Minimum Cut (VieCut)](#global-minimum-cut-viecut)
  - [Community Detection (VieClus)](#community-detection-vieclus)
  - [Maximum Weight Independent Set (CHILS)](#maximum-weight-independent-set-chils)
  - [Maximum Independent Set (KaMIS)](#maximum-independent-set-kamis)
  - [Correlation Clustering (SCC)](#correlation-clustering-scc)
  - [Edge Orientation (HeiOrient)](#edge-orientation-heiorient)
  - [Streaming Graph Partitioning (HeiStream)](#streaming-graph-partitioning-heistream)
  - [Longest Path (KaLP)](#longest-path-kalp)
  - [Maximum Cut (fpt-max-cut)](#maximum-cut-fpt-max-cut)
  - [Local Motif Clustering](#local-motif-clustering)
- [Use Cases & Examples](#use-cases--examples)
  - [Distributed Computing: Domain Decomposition](#distributed-computing-domain-decomposition)
  - [Social Network Analysis](#social-network-analysis)
  - [Algorithm Benchmarking](#algorithm-benchmarking)
  - [Sparse Linear Algebra](#sparse-linear-algebra)
  - [Streaming & Dynamic Graphs](#streaming--dynamic-graphs)
- [I/O](#io)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

---

## Why CHSZLabLib?

Graph algorithm research produces highly optimized C/C++ implementations that are difficult to use from Python: incompatible data formats, complex build systems, and no shared graph representation. Switching between libraries means converting data, learning new APIs, and managing separate installations.

**CHSZLabLib solves this.** It wraps 11 established algorithm libraries behind a single `Graph` object backed by CSR-format NumPy arrays. Every algorithm reads from the same in-memory representation with zero data copying.

```python
from chszlablib import Graph, partition, mincut, cluster

g = Graph.from_metis("social_network.graph")

# Three different algorithms, one graph, zero copies
p  = partition(g, num_parts=8, mode="strong")
mc = mincut(g, algorithm="viecut")
c  = cluster(g, time_limit=5.0)
```

**Key properties:**

- **Unified API** — consistent function signatures, dataclass results, NumPy arrays throughout
- **Zero-copy sharing** — all algorithms operate on the same CSR arrays; no serialization overhead
- **C/C++ performance** — every algorithm runs as compiled native code via pybind11
- **Pythonic design** — builder pattern for graph construction, lazy finalization, keyword arguments with sensible defaults

---

## Integrated Libraries

| Library | Domain | Algorithms |
|:--------|:-------|:-----------|
| [KaHIP](https://github.com/KaHIP/KaHIP) | Graph partitioning | KaFFPa (6 modes), KaFFPaE (evolutionary), node separators, nested dissection |
| [VieCut](https://github.com/VieCut/VieCut) | Minimum cuts | VieCut, NOI, Karger-Stein, Matula, Padberg-Rinaldi, Cactus |
| [VieClus](https://github.com/VieClus/VieClus) | Community detection | Modularity-maximizing evolutionary clustering |
| [CHILS](https://github.com/KarlsruheMIS/CHILS) | Weighted independent set | Concurrent heuristic independent local search |
| [KaMIS](https://github.com/KarlsruheMIS/KaMIS) | Independent set | ReduMIS, OnlineMIS, Branch&Reduce, MMWIS |
| [SCC](https://github.com/ScalableCorrelationClustering/ScalableCorrelationClustering) | Correlation clustering | Label propagation + evolutionary on signed graphs |
| [HeiOrient](https://github.com/HeiOrient/HeiOrient) | Edge orientation | 2-approx greedy, DFS local search, Eager Path Search |
| [HeiStream](https://github.com/KaHIP/HeiStream) | Streaming partitioning | Fennel, BuffCut, parallel pipeline, batched model |
| [KaLP](https://github.com/schulzchristian/KaLP) | Longest paths | Partitioning-aided longest simple path solver |
| [fpt-max-cut](https://github.com/KarlsruheMIS/fpt-max-cut) | Maximum cut | FPT kernelization + heuristic/exact solvers |
| [HeidelbergMotifClustering](https://github.com/schulzchristian/HeidelbergMotifClustering) | Local clustering | Triangle-motif-based flow and partitioning methods |

---

## Quick Start

```python
from chszlablib import (
    Graph, partition, mincut, cluster, mwis, redumis,
    orient_edges, stream_partition, longest_path, maxcut, motif_cluster,
)

# Build a small graph
g = Graph(num_nodes=6)
for u, v in [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (2,3)]:
    g.add_edge(u, v)

# Partition into 2 balanced blocks
p = partition(g, num_parts=2, mode="strong")
print(f"Edge cut: {p.edgecut}, assignment: {p.assignment}")

# Global minimum cut
mc = mincut(g, algorithm="viecut")
print(f"Min-cut value: {mc.cut_value}")

# Community detection
c = cluster(g, time_limit=1.0)
print(f"Modularity: {c.modularity:.4f}, clusters: {c.num_clusters}")

# Edge orientation
eo = orient_edges(g, algorithm="combined")
print(f"Max out-degree: {eo.max_out_degree}")

# Streaming partitioning
sp = stream_partition(g, k=3, imbalance=3.0)
print(f"Stream assignment: {sp.assignment}")
```

---

## Installation

```bash
git clone --recursive https://github.com/CHSZLab/CHSZLabLib.git
cd CHSZLabLib
bash build.sh
```

The build script handles everything automatically:

1. Initializes and updates all Git submodules
2. Creates a Python virtual environment in `.venv/`
3. Compiles all C/C++ extensions via CMake + pybind11
4. Builds and installs the Python wheel
5. Runs the full test suite

**Requirements:**

| Dependency | Version |
|:-----------|:--------|
| Python | >= 3.9 |
| C++ compiler | GCC or Clang with C++17 support |
| CMake | >= 3.15 |
| NumPy | >= 1.20 |
| OpenMP | Optional (enables parallelism in VieClus, CHILS, HeiStream) |

---

## Graph Construction

All algorithms operate on the `Graph` class, which stores data in Compressed Sparse Row (CSR) format. Three construction methods are available:

### Builder API (edge-by-edge)

```python
from chszlablib import Graph

g = Graph(num_nodes=5)
g.add_edge(0, 1, weight=3)
g.add_edge(1, 2)
g.add_edge(2, 3, weight=7)
g.add_edge(3, 4)
g.set_node_weight(0, 10)
g.finalize()  # converts to CSR; auto-called on first property access

print(g.num_nodes)     # 5
print(g.num_edges)     # 4
print(g.xadj)          # CSR row pointers
print(g.adjncy)        # CSR column indices
print(g.edge_weights)  # per-entry edge weights
print(g.node_weights)  # per-node weights
```

### Direct CSR construction

For large graphs or interoperability with SciPy/NetworkX:

```python
import numpy as np
from chszlablib import Graph

# 4-node cycle: 0-1-2-3-0
g = Graph.from_csr(
    xadj    = np.array([0, 2, 4, 6, 8]),
    adjncy  = np.array([1, 3, 0, 2, 1, 3, 0, 2]),
    edge_weights = np.array([1, 2, 1, 3, 3, 5, 2, 5]),  # optional
)
```

### From METIS file

```python
from chszlablib import Graph, read_metis

# Class method
g = Graph.from_metis("mesh.graph")

# Module function (equivalent)
g = read_metis("mesh.graph")

# Write back
g.to_metis("output.graph")
```

### Converting from NetworkX

```python
import networkx as nx
import numpy as np
from chszlablib import Graph

G_nx = nx.karate_club_graph()
A = nx.to_scipy_sparse_array(G_nx, format="csr")

g = Graph.from_csr(
    xadj   = np.array(A.indptr, dtype=np.int64),
    adjncy = np.array(A.indices, dtype=np.int32),
)
```

---

## Algorithms & API Reference

Every algorithm function follows the same pattern: it takes a `Graph` and configuration parameters, and returns a typed dataclass with results as NumPy arrays.

---

### Graph Partitioning (KaHIP)

Partition a graph into *k* balanced blocks, minimizing the number of edges between blocks.

```python
partition(g, num_parts=2, mode="eco", imbalance=0.03, seed=0) -> PartitionResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `num_parts` | `int` | `2` | Number of blocks |
| `mode` | `str` | `"eco"` | Quality/speed trade-off (see table below) |
| `imbalance` | `float` | `0.03` | Allowed weight imbalance (0.03 = 3%) |
| `seed` | `int` | `0` | Random seed for reproducibility |

**Partitioning modes:**

| Mode | Speed | Quality | Best for |
|:-----|:------|:--------|:---------|
| `"fast"` | Fastest | Good | Large-scale exploration |
| `"eco"` | Balanced | Very good | Default choice |
| `"strong"` | Slowest | Best | Final production partitions |
| `"fastsocial"` | Fastest | Good | Social / power-law networks |
| `"ecosocial"` | Balanced | Very good | Social / power-law networks |
| `"strongsocial"` | Slowest | Best | Social / power-law networks |

**Result: `PartitionResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `edgecut` | `int` | Number of edges crossing block boundaries |
| `assignment` | `ndarray[int32]` | Block ID for each node (0 to *k*-1) |

```python
from chszlablib import Graph, partition

g = Graph.from_metis("mesh.graph")

# Quick exploration
p_fast = partition(g, num_parts=8, mode="fast")

# Production quality
p_strong = partition(g, num_parts=8, mode="strong", imbalance=0.01)

print(f"Fast edgecut:   {p_fast.edgecut}")
print(f"Strong edgecut: {p_strong.edgecut}")
```

#### Node Separator

Find a small set of nodes whose removal disconnects the graph.

```python
node_separator(g, num_parts=2, mode="eco", imbalance=0.03, seed=0) -> SeparatorResult
```

**Result:** `SeparatorResult` with `num_separator_vertices` (int) and `separator` (ndarray of node indices).

```python
from chszlablib import node_separator

sep = node_separator(g, mode="strong")
print(f"Separator size: {sep.num_separator_vertices}")
print(f"Separator nodes: {sep.separator}")
```

#### Nested Dissection Ordering

Compute a fill-reducing ordering for sparse matrix factorization.

```python
node_ordering(g, mode="eco", seed=0) -> OrderingResult
```

**Result:** `OrderingResult` with `ordering` (ndarray permutation of length *n*).

```python
from chszlablib import node_ordering

order = node_ordering(g, mode="strong")
# Use order.ordering as a permutation for sparse Cholesky factorization
```

---

### Evolutionary Partitioning (KaHIP)

KaFFPaE runs a memetic (evolutionary) algorithm that iteratively improves partition quality. It supports warm-starting from an existing partition.

```python
kaffpaE(g, num_parts, time_limit, mode="strong", imbalance=0.03, seed=0,
        initial_partition=None) -> PartitionResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `num_parts` | `int` | *required* | Number of blocks |
| `time_limit` | `int` | *required* | Seconds for the evolutionary search |
| `mode` | `str` | `"strong"` | Quality preset (includes `"ultrafastsocial"`) |
| `imbalance` | `float` | `0.03` | Allowed imbalance |
| `initial_partition` | `ndarray` | `None` | Warm-start from an existing assignment |

**Result: `PartitionResult`** with `edgecut`, `assignment`, and `balance`.

```python
from chszlablib import partition, kaffpaE

g = Graph.from_metis("large_mesh.graph")

# Two-phase approach: quick seed, then refine
seed_part = partition(g, num_parts=16, mode="eco")
refined   = kaffpaE(g, num_parts=16, time_limit=60,
                    initial_partition=seed_part.assignment)

print(f"Seed edgecut:    {seed_part.edgecut}")
print(f"Refined edgecut: {refined.edgecut}")
print(f"Balance:         {refined.balance:.4f}")
```

---

### Global Minimum Cut (VieCut)

Compute the global minimum cut of an undirected graph — the smallest set of edges whose removal disconnects the graph.

```python
mincut(g, algorithm="viecut", seed=0) -> MincutResult
```

| Algorithm | Identifier | Characteristics |
|:----------|:-----------|:----------------|
| VieCut | `"viecut"` or `"vc"` | Near-linear time; best for large graphs |
| NOI | `"noi"` | Deterministic; Nagamochi-Ono-Ibaraki |
| Karger-Stein | `"ks"` | Randomized; Monte Carlo approach |
| Matula | `"matula"` | Approximation-based |
| Padberg-Rinaldi | `"pr"` | Exact; LP-based heuristic |
| Cactus | `"cactus"` | Enumerates all minimum cuts |

**Result: `MincutResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `cut_value` | `int` | Weight of the minimum cut |
| `partition` | `ndarray[int32]` | 0/1 assignment per node |

```python
from chszlablib import mincut

mc = mincut(g, algorithm="viecut")
side_a = (mc.partition == 0).sum()
side_b = (mc.partition == 1).sum()
print(f"Min-cut: {mc.cut_value} (splits into {side_a} and {side_b} nodes)")
```

---

### Community Detection (VieClus)

Detect communities by maximizing modularity using a population-based evolutionary approach.

```python
cluster(g, time_limit=1.0, seed=0, cluster_upperbound=0) -> ClusterResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `time_limit` | `float` | `1.0` | Seconds for optimization |
| `cluster_upperbound` | `int` | `0` | Maximum cluster size (0 = unlimited) |

**Result: `ClusterResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `modularity` | `float` | Modularity score (higher is better) |
| `num_clusters` | `int` | Number of detected communities |
| `assignment` | `ndarray[int32]` | Cluster ID for each node |

```python
from chszlablib import cluster

c = cluster(g, time_limit=10.0)
print(f"Found {c.num_clusters} communities, modularity = {c.modularity:.4f}")

# Constrain maximum community size
c2 = cluster(g, time_limit=10.0, cluster_upperbound=100)
```

---

### Maximum Weight Independent Set (CHILS)

Find a set of non-adjacent nodes with maximum total weight using concurrent parallel heuristic search.

```python
mwis(g, time_limit=10.0, num_concurrent=4, seed=0) -> MWISResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `time_limit` | `float` | `10.0` | Seconds for optimization |
| `num_concurrent` | `int` | `4` | Parallel solution attempts |

**Result: `MWISResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `weight` | `int` | Total weight of the independent set |
| `vertices` | `ndarray[int32]` | Node indices in the solution |

```python
from chszlablib import Graph, mwis

g = Graph(num_nodes=5)
for u, v in [(0,1), (1,2), (2,3), (3,4)]:
    g.add_edge(u, v)
for i in range(5):
    g.set_node_weight(i, (i + 1) * 10)  # weights: 10, 20, 30, 40, 50

result = mwis(g, time_limit=5.0, num_concurrent=8)
print(f"Weight: {result.weight}, vertices: {result.vertices}")
```

---

### Maximum Independent Set (KaMIS)

Four algorithms for finding maximum independent sets, covering both unweighted and weighted variants. All return `MISResult` with `size`, `weight`, and `vertices`.

#### Unweighted MIS

**ReduMIS** — evolutionary algorithm with graph reductions. Best solution quality.

```python
redumis(g, time_limit=10.0, seed=0, full_kernelization=False) -> MISResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `time_limit` | `float` | `10.0` | Seconds for optimization |
| `full_kernelization` | `bool` | `False` | `True` for full kernelization, `False` for FastKer |

**OnlineMIS** — iterated local search. Faster, useful for quick approximations.

```python
online_mis(g, time_limit=10.0, seed=0, ils_iterations=15000) -> MISResult
```

#### Weighted MIS

**Branch & Reduce** — exact solver with data reduction rules.

```python
branch_reduce(g, time_limit=10.0, seed=0) -> MISResult
```

**MMWIS** — memetic evolutionary algorithm. Trades exactness for scalability.

```python
mmwis_solver(g, time_limit=10.0, seed=0) -> MISResult
```

```python
from chszlablib import redumis, online_mis, branch_reduce

# Compare unweighted algorithms
r1 = redumis(g, time_limit=5.0)
r2 = online_mis(g, time_limit=5.0, ils_iterations=20000)
print(f"ReduMIS: {r1.size} nodes | OnlineMIS: {r2.size} nodes")

# Exact weighted solver
r3 = branch_reduce(g, time_limit=30.0)
print(f"Exact MWIS weight: {r3.weight}")
```

---

### Correlation Clustering (SCC)

Cluster a signed graph by minimizing edge disagreements. Positive edge weights attract nodes into the same cluster; negative weights repel them into different clusters.

```python
correlation_clustering(g, seed=0, time_limit=0) -> CorrelationClusteringResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `time_limit` | `float` | `0` | 0 = single run; > 0 = repeat and keep best |

```python
evolutionary_correlation_clustering(g, seed=0, time_limit=5.0) -> CorrelationClusteringResult
```

Population-based evolutionary variant — generally better quality at the cost of runtime.

**Result: `CorrelationClusteringResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `edge_cut` | `int` | Number of disagreement edges |
| `num_clusters` | `int` | Number of clusters found |
| `assignment` | `ndarray[int32]` | Cluster ID for each node |

```python
from chszlablib import Graph, correlation_clustering, evolutionary_correlation_clustering

# Signed graph: friends and foes
g = Graph(num_nodes=6)
g.add_edge(0, 1, weight=1)    # friends
g.add_edge(1, 2, weight=1)    # friends
g.add_edge(0, 2, weight=1)    # friends
g.add_edge(3, 4, weight=1)    # friends
g.add_edge(4, 5, weight=1)    # friends
g.add_edge(3, 5, weight=1)    # friends
g.add_edge(2, 3, weight=-1)   # foes — should be in different clusters

cc = correlation_clustering(g)
print(f"Clusters: {cc.num_clusters}, disagreements: {cc.edge_cut}")

# Higher quality with evolutionary approach
cc2 = evolutionary_correlation_clustering(g, time_limit=5.0)
print(f"Evo clusters: {cc2.num_clusters}, disagreements: {cc2.edge_cut}")
```

---

### Edge Orientation (HeiOrient)

Orient all undirected edges to minimize the maximum out-degree across all nodes.

```python
orient_edges(g, algorithm="combined", seed=0, eager_size=100) -> EdgeOrientationResult
```

| Algorithm | Identifier | Characteristics |
|:----------|:-----------|:----------------|
| 2-Approximation | `"two_approx"` | Fast greedy; guaranteed 2-approximation |
| DFS Local Search | `"dfs"` | DFS-based improvement |
| Eager Path Search | `"combined"` | Best quality; combines both approaches |

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `algorithm` | `str` | `"combined"` | Algorithm to use |
| `eager_size` | `int` | `100` | Eager threshold for combined algorithm |

**Result: `EdgeOrientationResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `max_out_degree` | `int` | Maximum out-degree achieved |
| `out_degrees` | `ndarray[int32]` | Per-node out-degree |
| `edge_heads` | `ndarray[int32]` | 0/1 per CSR entry: 1 = oriented away from row node |

```python
from chszlablib import orient_edges

for algo in ["two_approx", "dfs", "combined"]:
    r = orient_edges(g, algorithm=algo)
    print(f"{algo:12s} -> max out-degree = {r.max_out_degree}")
```

---

### Streaming Graph Partitioning (HeiStream)

Partition a graph using streaming algorithms that process nodes sequentially, requiring far less memory than full in-memory partitioning.

```python
stream_partition(g, k=2, imbalance=3.0, seed=0, max_buffer_size=0,
                 batch_size=0, num_streams_passes=1,
                 run_parallel=False) -> StreamPartitionResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `k` | `int` | `2` | Number of partitions |
| `imbalance` | `float` | `3.0` | Allowed imbalance (%) |
| `max_buffer_size` | `int` | `0` | Buffer size for BuffCut (0 = default, 1 = no buffer) |
| `batch_size` | `int` | `0` | Batch size for model partitioning (0 = 16384) |
| `num_streams_passes` | `int` | `1` | Number of streaming passes (restreaming) |
| `run_parallel` | `bool` | `False` | Enable 3-thread parallel pipeline |

**Execution modes** (selected automatically by parameter combination):

| Mode | Trigger | Description |
|:-----|:--------|:------------|
| Direct Fennel | `max_buffer_size=1, batch_size=1` | Fastest; one-pass Fennel scoring |
| BuffCut | `max_buffer_size > 1` | Priority-buffered multi-level partitioning |
| BuffCut parallel | `max_buffer_size > 1, run_parallel=True` | 3-thread pipeline (I/O + queue + partitioner) |
| Batched (default) | default parameters | Batched model partitioning via KaHIP |

```python
from chszlablib import stream_partition

# Fast one-pass streaming
sp = stream_partition(g, k=4, max_buffer_size=1, batch_size=1)

# High-quality buffered partitioning
sp = stream_partition(g, k=4, max_buffer_size=1000, batch_size=100)

# Parallel pipeline for throughput
sp = stream_partition(g, k=4, max_buffer_size=1000, batch_size=100, run_parallel=True)

# Multiple passes for better quality
sp = stream_partition(g, k=4, max_buffer_size=1000, num_streams_passes=3)
```

#### Incremental Streaming API

For true streaming scenarios where the graph arrives node-by-node:

```python
from chszlablib import HeiStreamPartitioner

hs = HeiStreamPartitioner(k=4, imbalance=3.0, max_buffer_size=1000)

# Stream nodes as they arrive
hs.new_node(0, [1, 2])
hs.new_node(1, [0, 3])
hs.new_node(2, [0])
hs.new_node(3, [1])

result = hs.partition()
print(result.assignment)

# Reset and reuse for a different graph
hs.reset()
```

---

### Longest Path (KaLP)

Compute the longest simple path between two vertices. Uses graph partitioning to decompose the problem into smaller subproblems.

```python
longest_path(g, start_vertex=0, target_vertex=-1, partition_config="eco",
             block_size=10, number_of_threads=1, split_steps=0,
             threshold=0) -> LongestPathResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `start_vertex` | `int` | `0` | Start node |
| `target_vertex` | `int` | `-1` | Target node (-1 = last node) |
| `partition_config` | `str` | `"eco"` | Partitioning quality (`"fast"`, `"eco"`, `"strong"`) |
| `block_size` | `int` | `10` | Block size for partitioning step |
| `number_of_threads` | `int` | `1` | Thread count |

**Result: `LongestPathResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `length` | `int` | Path length (0 if no path exists) |
| `path` | `ndarray[int32]` | Ordered sequence of vertices |

```python
from chszlablib import Graph, longest_path

# Simple chain with a shortcut
g = Graph(num_nodes=6)
for u, v in [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5)]:
    g.add_edge(u, v)

result = longest_path(g, start_vertex=0, target_vertex=5)
print(f"Longest path length: {result.length}")
print(f"Path: {result.path}")
# Expected: length 5 via 0 -> 1 -> 2 -> 3 -> 4 -> 5
```

---

### Maximum Cut (fpt-max-cut)

Compute a maximum cut — a partition of nodes into two sets that maximizes the total weight of edges between the sets. Uses FPT data-reduction rules (kernelization) before solving.

```python
maxcut(g, method="heuristic", time_limit=1.0) -> MaxCutResult
```

| Method | Identifier | Characteristics |
|:-------|:-----------|:----------------|
| Heuristic | `"heuristic"` | Fast; good for large graphs |
| Exact | `"exact"` | FPT algorithm; feasible when kernelization reduces the instance sufficiently |

**Result: `MaxCutResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `cut_value` | `int` | Total weight of edges crossing the cut |
| `partition` | `ndarray[int32]` | 0/1 assignment per node |

```python
from chszlablib import maxcut

# Fast heuristic
mc = maxcut(g, method="heuristic", time_limit=5.0)
print(f"Max-cut (heuristic): {mc.cut_value}")

# Exact solver (small instances or well-kernelizable graphs)
mc_exact = maxcut(g, method="exact", time_limit=60.0)
print(f"Max-cut (exact):     {mc_exact.cut_value}")
```

---

### Local Motif Clustering

Find a local cluster around a seed node based on triangle motifs. Useful for discovering tightly-knit neighborhoods in large networks without processing the entire graph.

```python
motif_cluster(g, seed_node, method="social", bfs_depths=None,
              time_limit=60, seed=0) -> MotifClusterResult
```

| Method | Identifier | Characteristics |
|:-------|:-----------|:----------------|
| SOCIAL | `"social"` | Flow-based; faster (ESA 2023) |
| LMCHGP | `"lmchgp"` | Graph-partitioning-based (ALENEX 2023) |

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `seed_node` | `int` | *required* | Center node for local clustering |
| `bfs_depths` | `list[int]` | `[10, 15, 20]` | BFS neighborhood depths to try |
| `time_limit` | `int` | `60` | Seconds limit |

**Result: `MotifClusterResult`**

| Field | Type | Description |
|:------|:-----|:------------|
| `cluster_nodes` | `ndarray[int32]` | Node IDs in the discovered cluster |
| `motif_conductance` | `float` | Motif conductance score (lower = better) |

```python
from chszlablib import motif_cluster

# Find the local community around node 42
result = motif_cluster(g, seed_node=42, method="social")
print(f"Cluster size: {len(result.cluster_nodes)}")
print(f"Motif conductance: {result.motif_conductance:.4f}")

# Try the partitioning-based method
result2 = motif_cluster(g, seed_node=42, method="lmchgp", bfs_depths=[5, 10])
```

---

## Use Cases & Examples

### Distributed Computing: Domain Decomposition

Partition a computational mesh across MPI ranks to minimize inter-process communication:

```python
from chszlablib import Graph, partition, kaffpaE

# Load a finite element mesh
mesh = Graph.from_metis("engine_block.graph")
num_ranks = 64

# Phase 1: quick initial partition
initial = partition(mesh, num_parts=num_ranks, mode="eco")
print(f"Initial edgecut: {initial.edgecut:,}")

# Phase 2: evolutionary refinement
refined = kaffpaE(mesh, num_parts=num_ranks, time_limit=120,
                  initial_partition=initial.assignment)
print(f"Refined edgecut: {refined.edgecut:,} (balance: {refined.balance:.4f})")

# Export partition for MPI distribution
import numpy as np
np.save("rank_assignment.npy", refined.assignment)
```

### Social Network Analysis

Analyze community structure and influential nodes in a social network:

```python
from chszlablib import (
    Graph, cluster, mincut, mwis,
    correlation_clustering, motif_cluster,
)

g = Graph.from_metis("twitter_follows.graph")

# Detect communities
communities = cluster(g, time_limit=30.0)
print(f"Communities: {communities.num_clusters}")
print(f"Modularity:  {communities.modularity:.4f}")

# Find the most weakly connected region
mc = mincut(g, algorithm="viecut")
print(f"Network bottleneck: {mc.cut_value} edges")

# Explore a specific user's neighborhood
local = motif_cluster(g, seed_node=1337, method="social")
print(f"User 1337's tight community: {len(local.cluster_nodes)} members")
print(f"Motif conductance: {local.motif_conductance:.4f}")

# Weighted independent set for influence maximization
# Weight nodes by follower count
for node in range(g.num_nodes):
    g.set_node_weight(node, follower_counts[node])
influencers = mwis(g, time_limit=30.0, num_concurrent=8)
print(f"Selected {len(influencers.vertices)} non-adjacent influencers")
print(f"Total reach: {influencers.weight:,}")
```

### Algorithm Benchmarking

Systematically compare algorithm configurations on the same graph:

```python
from chszlablib import Graph, partition, mincut
import time

g = Graph.from_metis("benchmark_graph.graph")
print(f"Graph: {g.num_nodes:,} nodes, {g.num_edges:,} edges\n")

# Benchmark partition modes
print("Partitioning (k=16):")
print(f"  {'Mode':<15} {'Edgecut':>10} {'Time (s)':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10}")
for mode in ["fast", "eco", "strong", "fastsocial", "ecosocial", "strongsocial"]:
    t0 = time.perf_counter()
    r = partition(g, num_parts=16, mode=mode)
    dt = time.perf_counter() - t0
    print(f"  {mode:<15} {r.edgecut:>10,} {dt:>10.3f}")

# Benchmark mincut algorithms
print("\nMinimum Cut:")
print(f"  {'Algorithm':<15} {'Cut Value':>10} {'Time (s)':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10}")
for algo in ["viecut", "noi", "ks", "matula", "pr"]:
    t0 = time.perf_counter()
    r = mincut(g, algorithm=algo)
    dt = time.perf_counter() - t0
    print(f"  {algo:<15} {r.cut_value:>10,} {dt:>10.3f}")
```

### Sparse Linear Algebra

Use nested dissection ordering to reduce fill-in during sparse matrix factorization:

```python
from chszlablib import Graph, node_ordering
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import splu

# Build graph from sparse matrix structure
A = csr_array(load_sparse_matrix())
g = Graph.from_csr(
    xadj   = np.array(A.indptr, dtype=np.int64),
    adjncy = np.array(A.indices, dtype=np.int32),
)

# Compute fill-reducing permutation
order = node_ordering(g, mode="strong")
perm = order.ordering

# Apply permutation to matrix
A_reordered = A[perm][:, perm]

# Factor with reduced fill-in
lu = splu(A_reordered.tocsc())
print(f"Non-zeros in L+U: {lu.nnz:,}")
```

### Streaming & Dynamic Graphs

Process graphs that are too large for memory or arrive incrementally:

```python
from chszlablib import HeiStreamPartitioner

# Configure streaming partitioner
partitioner = HeiStreamPartitioner(
    k=32,
    imbalance=3.0,
    max_buffer_size=10000,
    batch_size=1000,
)

# Simulate streaming edge data
with open("edge_stream.txt") as f:
    current_node = 0
    neighbors = []
    for line in f:
        src, dst = map(int, line.split())
        if src != current_node:
            partitioner.new_node(current_node, neighbors)
            current_node = src
            neighbors = []
        neighbors.append(dst)
    partitioner.new_node(current_node, neighbors)  # last node

result = partitioner.partition()
print(f"Assigned {len(result.assignment)} nodes to {32} partitions")

# Process next batch with a fresh partitioner state
partitioner.reset()
```

---

## I/O

Read and write graphs in [METIS format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf):

```python
from chszlablib import read_metis, write_metis

g = read_metis("input.graph")
write_metis(g, "output.graph")

# Equivalent via Graph methods
g = Graph.from_metis("input.graph")
g.to_metis("output.graph")
```

Supports unweighted, edge-weighted, node-weighted, and fully weighted graphs.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

The test suite covers all algorithm families with correctness checks, edge cases, and cross-algorithm integration tests.

---

## Project Structure

```
CHSZLabLib/
├── chszlablib/                  # Python package
│   ├── __init__.py              # Public API exports
│   ├── graph.py                 # Graph class (CSR backend)
│   ├── partition.py             # KaHIP partitioning
│   ├── mincut.py                # VieCut minimum cuts
│   ├── cluster.py               # VieClus community detection
│   ├── mwis.py                  # CHILS weighted independent set
│   ├── mis.py                   # KaMIS independent set algorithms
│   ├── correlation_clustering.py# SCC signed graph clustering
│   ├── orientation.py           # HeiOrient edge orientation
│   ├── heistream.py             # HeiStream streaming partitioning
│   ├── longest_path.py          # KaLP longest paths
│   ├── maxcut.py                # fpt-max-cut maximum cut
│   ├── motif_clustering.py      # Local motif clustering
│   └── io.py                    # METIS file I/O
├── bindings/                    # pybind11 C++ bindings (15 modules)
├── tests/                       # pytest suite (18 test files)
├── KaHIP/                       # Submodule: graph partitioning
├── VieCut/                      # Submodule: minimum cuts
├── VieClus/                     # Submodule: clustering
├── CHILS/                       # Submodule: weighted independent set
├── KaMIS/                       # Submodule: independent set algorithms
├── SCC/                         # Submodule: correlation clustering
├── HeiOrient/                   # Submodule: edge orientation
├── HeiStream/                   # Submodule: streaming partitioning
├── KaLP/                        # Submodule: longest paths
├── fpt-max-cut/                 # Submodule: maximum cut
├── HeidelbergMotifClustering/   # Submodule: motif clustering
├── CMakeLists.txt               # Top-level CMake configuration
├── pyproject.toml               # Python package metadata
├── build.sh                     # One-step build script
└── demo.py                      # Full demonstration script
```

---

<p align="center">
  <sub>MIT License</sub>
</p>
