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

## About

CHSZLabLib is a **usability-focused** Python wrapper around 11 high-performance C/C++ graph algorithm libraries. It is designed for researchers, practitioners, and AI agents who want easy access to state-of-the-art graph algorithms through a clean, consistent API.

**For maximum performance** (custom parameter tuning, MPI-level parallelism, full algorithmic control), use the underlying C/C++ repositories directly. This library prioritizes convenience and a unified interface over exposing every possible knob.

---

## Table of Contents

- [About](#about)
- [Integrated Libraries](#integrated-libraries)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Graph Construction](#graph-construction)
- [API Reference](#api-reference)
  - [Decomposition](#decomposition)
  - [IndependenceProblems](#independenceproblems)
  - [Orientation](#orientation)
  - [PathProblems](#pathproblems)
  - [HeiStreamPartitioner](#heistreampartitioner)
- [Use Cases & Examples](#use-cases--examples)
- [I/O](#io)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Citations](#citations)
- [Authors & Acknowledgments](#authors--acknowledgments)

---

## Integrated Libraries

| Library | Domain | Algorithms |
|:--------|:-------|:-----------|
| [KaHIP](https://github.com/KaHIP/KaHIP) | Graph partitioning | KaFFPa (6 modes), KaFFPaE (evolutionary), node separators, nested dissection |
| [VieCut](https://github.com/VieCut/VieCut) | Minimum cuts | VieCut, NOI, Karger-Stein, Matula, Padberg-Rinaldi, Cactus |
| [VieClus](https://github.com/VieClus/VieClus) | Community detection | Modularity-maximizing evolutionary clustering |
| [CHILS](https://github.com/KennethLangedal/CHILS) | Weighted independent set | Concurrent heuristic independent local search |
| [KaMIS](https://github.com/KarlsruheMIS/KaMIS) | Independent set | ReduMIS, OnlineMIS, Branch&Reduce, MMWIS |
| [SCC](https://github.com/ScalableCorrelationClustering/ScalableCorrelationClustering) | Correlation clustering | Label propagation + evolutionary on signed graphs |
| [HeiOrient](https://github.com/KaHIP/HeiOrient) | Edge orientation | 2-approx greedy, DFS local search, Eager Path Search |
| [HeiStream](https://github.com/KaHIP/HeiStream) | Streaming partitioning | Fennel, BuffCut, parallel pipeline, batched model |
| [KaLP](https://github.com/KarlsruheLongestPaths/KaLP) | Longest paths | Partitioning-aided longest simple path solver |
| [fpt-max-cut](https://github.com/KarlsruheMIS/fpt-max-cut) | Maximum cut | FPT kernelization + heuristic/exact solvers |
| [HeidelbergMotifClustering](https://github.com/LocalClustering/HeidelbergMotifClustering) | Local clustering | Triangle-motif-based flow and partitioning methods |

---

## Quick Start

```python
from chszlablib import Graph, Decomposition, IndependenceProblems, Orientation, PathProblems

# Build a small graph
g = Graph(num_nodes=6)
for u, v in [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (2,3)]:
    g.add_edge(u, v)

# Partition into 2 balanced blocks
p = Decomposition.partition(g, num_parts=2, mode="strong")
print(f"Edge cut: {p.edgecut}, assignment: {p.assignment}")

# Global minimum cut
mc = Decomposition.mincut(g, algorithm="viecut")
print(f"Min-cut value: {mc.cut_value}")

# Community detection
c = Decomposition.cluster(g, time_limit=1.0)
print(f"Modularity: {c.modularity:.4f}, clusters: {c.num_clusters}")

# Edge orientation
eo = Orientation.orient_edges(g, algorithm="combined")
print(f"Max out-degree: {eo.max_out_degree}")

# Streaming partitioning
sp = Decomposition.stream_partition(g, k=3, imbalance=3.0)
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

## API Reference

The library organizes algorithms into four namespace classes. Each class is a pure namespace (no instantiation) with static methods. Every method takes a `Graph` and returns a typed dataclass with NumPy arrays.

---

### Decomposition

Graph decomposition: partitioning, cuts, clustering, and community detection.

#### `Decomposition.partition(g, ...)` — Graph Partitioning (KaHIP)

Partition a graph into *k* balanced blocks, minimizing the number of edges between blocks.

```python
Decomposition.partition(g, num_parts=2, mode="eco", imbalance=0.03, seed=0) -> PartitionResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `num_parts` | `int` | `2` | Number of blocks |
| `mode` | `str` | `"eco"` | Quality/speed trade-off |
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

**Result: `PartitionResult`** — `edgecut` (int), `assignment` (ndarray).

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("mesh.graph")
p = Decomposition.partition(g, num_parts=8, mode="strong", imbalance=0.01)
print(f"Edgecut: {p.edgecut}")
```

#### `Decomposition.evolutionary_partition(g, ...)` — Evolutionary Partitioning (KaHIP)

KaFFPaE runs a memetic (evolutionary) algorithm that iteratively improves partition quality. Supports warm-starting from an existing partition.

```python
Decomposition.evolutionary_partition(g, num_parts, time_limit, mode="strong",
                                     imbalance=0.03, seed=0,
                                     initial_partition=None) -> PartitionResult
```

**Result: `PartitionResult`** with `edgecut`, `assignment`, and `balance`.

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("large_mesh.graph")
seed_part = Decomposition.partition(g, num_parts=16, mode="eco")
refined = Decomposition.evolutionary_partition(
    g, num_parts=16, time_limit=60,
    initial_partition=seed_part.assignment,
)
print(f"Refined edgecut: {refined.edgecut} (balance: {refined.balance:.4f})")
```

#### `Decomposition.node_separator(g, ...)` — Node Separator (KaHIP)

Find a small set of nodes whose removal disconnects the graph.

```python
Decomposition.node_separator(g, num_parts=2, mode="eco", imbalance=0.03, seed=0) -> SeparatorResult
```

**Result: `SeparatorResult`** — `num_separator_vertices` (int), `separator` (ndarray).

#### `Decomposition.node_ordering(g, ...)` — Nested Dissection Ordering (KaHIP)

Compute a fill-reducing ordering for sparse matrix factorization.

```python
Decomposition.node_ordering(g, mode="eco", seed=0) -> OrderingResult
```

**Result: `OrderingResult`** — `ordering` (ndarray permutation).

#### `Decomposition.mincut(g, ...)` — Global Minimum Cut (VieCut)

```python
Decomposition.mincut(g, algorithm="viecut", seed=0) -> MincutResult
```

| Algorithm | Identifier | Characteristics |
|:----------|:-----------|:----------------|
| VieCut | `"viecut"` or `"vc"` | Near-linear time; best for large graphs |
| NOI | `"noi"` | Deterministic; Nagamochi-Ono-Ibaraki |
| Karger-Stein | `"ks"` | Randomized; Monte Carlo approach |
| Matula | `"matula"` | Approximation-based |
| Padberg-Rinaldi | `"pr"` | Exact; LP-based heuristic |
| Cactus | `"cactus"` | Enumerates all minimum cuts |

**Result: `MincutResult`** — `cut_value` (int), `partition` (ndarray 0/1).

#### `Decomposition.cluster(g, ...)` — Community Detection (VieClus)

```python
Decomposition.cluster(g, time_limit=1.0, seed=0, cluster_upperbound=0) -> ClusterResult
```

**Result: `ClusterResult`** — `modularity` (float), `num_clusters` (int), `assignment` (ndarray).

#### `Decomposition.maxcut(g, ...)` — Maximum Cut (fpt-max-cut)

```python
Decomposition.maxcut(g, method="heuristic", time_limit=1.0) -> MaxCutResult
```

| Method | Identifier | Characteristics |
|:-------|:-----------|:----------------|
| Heuristic | `"heuristic"` | Fast; good for large graphs |
| Exact | `"exact"` | FPT algorithm; feasible when kernelization reduces the instance sufficiently |

**Result: `MaxCutResult`** — `cut_value` (int), `partition` (ndarray 0/1).

#### `Decomposition.correlation_clustering(g, ...)` — Correlation Clustering (SCC)

Cluster a signed graph by minimizing edge disagreements.

```python
Decomposition.correlation_clustering(g, seed=0, time_limit=0) -> CorrelationClusteringResult
```

#### `Decomposition.evolutionary_correlation_clustering(g, ...)` — Evolutionary Correlation Clustering (SCC)

Population-based evolutionary variant.

```python
Decomposition.evolutionary_correlation_clustering(g, seed=0, time_limit=5.0) -> CorrelationClusteringResult
```

**Result: `CorrelationClusteringResult`** — `edge_cut` (int), `num_clusters` (int), `assignment` (ndarray).

#### `Decomposition.stream_partition(g, ...)` — Streaming Partitioning (HeiStream)

```python
Decomposition.stream_partition(g, k=2, imbalance=3.0, seed=0, max_buffer_size=0,
                               batch_size=0, num_streams_passes=1,
                               run_parallel=False) -> StreamPartitionResult
```

**Result: `StreamPartitionResult`** — `assignment` (ndarray).

#### `Decomposition.motif_cluster(g, ...)` — Local Motif Clustering

Find a local cluster around a seed node based on triangle motifs.

```python
Decomposition.motif_cluster(g, seed_node, method="social", bfs_depths=None,
                            time_limit=60, seed=0) -> MotifClusterResult
```

| Method | Identifier | Characteristics |
|:-------|:-----------|:----------------|
| SOCIAL | `"social"` | Flow-based; faster |
| LMCHGP | `"lmchgp"` | Graph-partitioning-based |

**Result: `MotifClusterResult`** — `cluster_nodes` (ndarray), `motif_conductance` (float).

---

### IndependenceProblems

Maximum independent set and maximum weight independent set solvers.

#### `IndependenceProblems.redumis(g, ...)` — ReduMIS (KaMIS)

Evolutionary algorithm with graph reductions. Best solution quality for unweighted MIS.

```python
IndependenceProblems.redumis(g, time_limit=10.0, seed=0, full_kernelization=False) -> MISResult
```

#### `IndependenceProblems.online_mis(g, ...)` — OnlineMIS (KaMIS)

Iterated local search. Faster, useful for quick approximations.

```python
IndependenceProblems.online_mis(g, time_limit=10.0, seed=0, ils_iterations=15000) -> MISResult
```

**Result: `MISResult`** — `size` (int), `weight` (int), `vertices` (ndarray).

#### `IndependenceProblems.branch_reduce(g, ...)` — Branch & Reduce (KaMIS)

Exact weighted MIS solver with data reduction rules.

```python
IndependenceProblems.branch_reduce(g, time_limit=10.0, seed=0) -> MISResult
```

#### `IndependenceProblems.mmwis(g, ...)` — MMWIS (KaMIS)

Memetic evolutionary algorithm for weighted MIS. Trades exactness for scalability.

```python
IndependenceProblems.mmwis(g, time_limit=10.0, seed=0) -> MISResult
```

#### `IndependenceProblems.chils(g, ...)` — CHILS

Concurrent heuristic independent local search for maximum weight independent set.

```python
IndependenceProblems.chils(g, time_limit=10.0, num_concurrent=4, seed=0) -> MWISResult
```

**Result: `MWISResult`** — `weight` (int), `vertices` (ndarray).

```python
from chszlablib import Graph, IndependenceProblems

g = Graph(num_nodes=5)
for u, v in [(0,1), (1,2), (2,3), (3,4)]:
    g.add_edge(u, v)
for i in range(5):
    g.set_node_weight(i, (i + 1) * 10)

result = IndependenceProblems.chils(g, time_limit=5.0, num_concurrent=8)
print(f"Weight: {result.weight}, vertices: {result.vertices}")
```

---

### Orientation

Edge orientation to minimize maximum out-degree.

#### `Orientation.orient_edges(g, ...)` — Edge Orientation (HeiOrient)

```python
Orientation.orient_edges(g, algorithm="combined", seed=0, eager_size=100) -> EdgeOrientationResult
```

| Algorithm | Identifier | Characteristics |
|:----------|:-----------|:----------------|
| 2-Approximation | `"two_approx"` | Fast greedy; guaranteed 2-approximation |
| DFS Local Search | `"dfs"` | DFS-based improvement |
| Eager Path Search | `"combined"` | Best quality; combines both approaches |

**Result: `EdgeOrientationResult`** — `max_out_degree` (int), `out_degrees` (ndarray), `edge_heads` (ndarray).

---

### PathProblems

Path-based graph problems.

#### `PathProblems.longest_path(g, ...)` — Longest Path (KaLP)

Compute the longest simple path between two vertices.

```python
PathProblems.longest_path(g, start_vertex=0, target_vertex=-1, partition_config="eco",
                          block_size=10, number_of_threads=1) -> LongestPathResult
```

**Result: `LongestPathResult`** — `length` (int), `path` (ndarray).

```python
from chszlablib import Graph, PathProblems

g = Graph(num_nodes=6)
for u, v in [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5)]:
    g.add_edge(u, v)

result = PathProblems.longest_path(g, start_vertex=0, target_vertex=5)
print(f"Longest path length: {result.length}, path: {result.path}")
```

---

### HeiStreamPartitioner

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

## Use Cases & Examples

### Distributed Computing: Domain Decomposition

```python
from chszlablib import Graph, Decomposition

mesh = Graph.from_metis("engine_block.graph")

# Phase 1: quick initial partition
initial = Decomposition.partition(mesh, num_parts=64, mode="eco")

# Phase 2: evolutionary refinement
refined = Decomposition.evolutionary_partition(
    mesh, num_parts=64, time_limit=120,
    initial_partition=initial.assignment,
)
print(f"Refined edgecut: {refined.edgecut:,} (balance: {refined.balance:.4f})")
```

### Social Network Analysis

```python
from chszlablib import Graph, Decomposition, IndependenceProblems

g = Graph.from_metis("twitter_follows.graph")

# Detect communities
communities = Decomposition.cluster(g, time_limit=30.0)
print(f"Communities: {communities.num_clusters}, modularity: {communities.modularity:.4f}")

# Find the most weakly connected region
mc = Decomposition.mincut(g, algorithm="viecut")
print(f"Network bottleneck: {mc.cut_value} edges")

# Explore a specific user's neighborhood
local = Decomposition.motif_cluster(g, seed_node=1337, method="social")
print(f"User 1337's tight community: {len(local.cluster_nodes)} members")

# Weighted independent set for influence maximization
result = IndependenceProblems.chils(g, time_limit=30.0, num_concurrent=8)
print(f"Selected {len(result.vertices)} non-adjacent influencers, total reach: {result.weight:,}")
```

### Sparse Linear Algebra

```python
from chszlablib import Graph, Decomposition
import numpy as np

# Compute fill-reducing permutation
order = Decomposition.node_ordering(g, mode="strong")
perm = order.ordering
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

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## Project Structure

```
CHSZLabLib/
├── chszlablib/                  # Python package
│   ├── __init__.py              # Public API exports
│   ├── graph.py                 # Graph class (CSR backend)
│   ├── decomposition.py         # Decomposition namespace (partitioning, cuts, clustering)
│   ├── independence.py          # IndependenceProblems namespace (MIS, MWIS)
│   ├── orientation.py           # Orientation namespace (edge orientation)
│   ├── paths.py                 # PathProblems namespace (longest path)
│   ├── heistream.py             # HeiStreamPartitioner (streaming)
│   └── io.py                    # METIS file I/O
├── bindings/                    # pybind11 C++ bindings
├── tests/                       # pytest suite
├── external_repositories/       # Git submodules (algorithm libraries)
│   ├── KaHIP/                   # Graph partitioning
│   ├── VieCut/                  # Minimum cuts
│   ├── VieClus/                 # Clustering
│   ├── CHILS/                   # Weighted independent set
│   ├── KaMIS/                   # Independent set algorithms
│   ├── SCC/                     # Correlation clustering
│   ├── HeiOrient/               # Edge orientation
│   ├── HeiStream/               # Streaming partitioning
│   ├── KaLP/                    # Longest paths
│   ├── fpt-max-cut/             # Maximum cut
│   └── HeidelbergMotifClustering/ # Motif clustering
├── CMakeLists.txt               # Top-level CMake configuration
├── pyproject.toml               # Python package metadata
├── build.sh                     # One-step build script
└── demo.py                      # Full demonstration script
```

---

## Citations

If you use CHSZLabLib in your research, please cite the relevant papers for each algorithm you use.

### KaHIP (Partitioning, Node Separators, Nested Dissection)

```bibtex
@inproceedings{sanders2013think,
  title     = {Think Locally, Act Globally: Highly Balanced Graph Partitioning},
  author    = {Peter Sanders and Christian Schulz},
  booktitle = {Proceedings of the 12th International Symposium on Experimental Algorithms (SEA'13)},
  series    = {LNCS},
  volume    = {7933},
  pages     = {164--175},
  year      = {2013},
  publisher = {Springer}
}

@article{meyerhenke2017parallel,
  title   = {Parallel Graph Partitioning for Complex Networks},
  author  = {Henning Meyerhenke and Peter Sanders and Christian Schulz},
  journal = {IEEE Transactions on Parallel and Distributed Systems},
  volume  = {28},
  number  = {9},
  pages   = {2625--2638},
  year    = {2017}
}
```

### VieCut (Minimum Cuts)

```bibtex
@article{henzinger2018practical,
  title   = {Practical Minimum Cut Algorithms},
  author  = {Monika Henzinger and Alexander Noe and Christian Schulz and Darren Strash},
  journal = {ACM Journal of Experimental Algorithmics},
  volume  = {23},
  year    = {2018}
}

@inproceedings{henzinger2020finding,
  title     = {Finding All Global Minimum Cuts in Practice},
  author    = {Monika Henzinger and Alexander Noe and Christian Schulz and Darren Strash},
  booktitle = {Proceedings of the 28th European Symposium on Algorithms (ESA'20)},
  year      = {2020}
}
```

### VieClus (Community Detection)

```bibtex
@inproceedings{biedermann2018memetic,
  title     = {Memetic Graph Clustering},
  author    = {Sonja Biedermann and Monika Henzinger and Christian Schulz and Bernhard Schuster},
  booktitle = {Proceedings of the 17th International Symposium on Experimental Algorithms (SEA'18)},
  series    = {LIPIcs},
  year      = {2018}
}
```

### CHILS (Weighted Independent Set)

```bibtex
@inproceedings{grossmann2025chils,
  title     = {Accelerating Reductions Using Graph Neural Networks and a New Concurrent Local Search for the Maximum Weight Independent Set Problem},
  author    = {Ernestine Gro{\ss}mann and Kenneth Langedal and Christian Schulz},
  booktitle = {Proceedings of the Symposium on Experimental Algorithms (SEA'25)},
  year      = {2025}
}
```

### KaMIS (Maximum Independent Set)

```bibtex
@article{lamm2017finding,
  title   = {Finding Near-Optimal Independent Sets at Scale},
  author  = {Sebastian Lamm and Peter Sanders and Christian Schulz and Darren Strash and Renato F. Werneck},
  journal = {Journal of Heuristics},
  volume  = {23},
  number  = {4},
  pages   = {207--229},
  year    = {2017}
}

@article{hespe2019scalable,
  title   = {Scalable Kernelization for Maximum Independent Sets},
  author  = {Demian Hespe and Christian Schulz and Darren Strash},
  journal = {ACM Journal of Experimental Algorithmics},
  volume  = {24},
  number  = {1},
  year    = {2019}
}

@inproceedings{lamm2019exactly,
  title     = {Exactly Solving the Maximum Weight Independent Set Problem on Large Real-World Graphs},
  author    = {Sebastian Lamm and Christian Schulz and Darren Strash and Robert Williger and Huashuo Zhang},
  booktitle = {Proceedings of ALENEX'19},
  pages     = {144--158},
  year      = {2019}
}

@inproceedings{grossmann2023mmwis,
  title     = {Finding Near-Optimal Weight Independent Sets at Scale},
  author    = {Ernestine Gro{\ss}mann and Sebastian Lamm and Christian Schulz and Darren Strash},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO'23)},
  pages     = {293--302},
  year      = {2023}
}
```

### SCC (Correlation Clustering)

```bibtex
@article{hausberger2024scalable,
  title   = {Scalable Multilevel and Memetic Signed Graph Clustering},
  author  = {Felix Hausberger and Marcelo Fonseca Faraj and Christian Schulz},
  journal = {arXiv preprint arXiv:2208.13618},
  year    = {2024}
}
```

### HeiStream (Streaming Partitioning)

```bibtex
@article{faraj2022buffered,
  title   = {Buffered Streaming Graph Partitioning},
  author  = {Marcelo Fonseca Faraj and Christian Schulz},
  journal = {ACM Journal of Experimental Algorithmics},
  year    = {2022},
  doi     = {10.1145/3546911}
}
```

### KaLP (Longest Path)

```bibtex
@inproceedings{fieger2019finding,
  title     = {Finding Optimal Longest Paths by Dynamic Programming in Parallel},
  author    = {Kai Fieger and Tom{\'a}s Balyo and Christian Schulz and Dominik Schreiber},
  booktitle = {Proceedings of the 12th Annual Symposium on Combinatorial Search (SOCS'19)},
  pages     = {61--69},
  year      = {2019},
  publisher = {AAAI Press}
}
```

### HeidelbergMotifClustering (Local Motif Clustering)

```bibtex
@inproceedings{chhabra2023local,
  title     = {Local Motif Clustering via (Hyper)Graph Partitioning},
  author    = {Adil Chhabra and Marcelo Fonseca Faraj and Christian Schulz},
  booktitle = {Proceedings of SIAM ALENEX'23},
  year      = {2023},
  doi       = {10.1137/1.9781611977561.ch9}
}

@inproceedings{chhabra2023faster,
  title     = {Faster Local Motif Clustering via Maximum Flows},
  author    = {Adil Chhabra and Marcelo Fonseca Faraj and Christian Schulz},
  booktitle = {Proceedings of the 31st European Symposium on Algorithms (ESA'23)},
  series    = {LIPIcs},
  year      = {2023}
}
```

---

## Authors & Acknowledgments

CHSZLabLib is developed by **Christian Schulz** at Heidelberg University.

This library would not be possible without the original algorithm implementations and research contributions from the following people:

- **Yaroslav Akhremtsev** — KaHIP
- **Tomás Balyo** — KaLP
- **Sonja Biedermann** — VieClus
- **Adil Chhabra** — HeiStream, HeidelbergMotifClustering, KaHIP
- **Jakob Dahlum** — KaMIS
- **Marcelo Fonseca Faraj** — SCC, HeiStream, HeidelbergMotifClustering, KaHIP
- **Kai Fieger** — KaLP
- **Alexander Gellner** — KaMIS
- **Ernestine Großmann** — CHILS, KaMIS (MMWIS)
- **Felix Hausberger** — SCC
- **Monika Henzinger** — VieCut, VieClus
- **Alexandra Henzinger** — KaHIP
- **Demian Hespe** — KaMIS
- **Sebastian Lamm** — KaMIS
- **Kenneth Langedal** — CHILS
- **Henning Meyerhenke** — KaHIP
- **Alexander Noe** — VieCut, KaHIP
- **Peter Sanders** — KaHIP, KaMIS
- **Sebastian Schlag** — KaHIP
- **Dominik Schreiber** — KaLP
- **Christian Schulz** — All libraries
- **Bernhard Schuster** — VieClus
- **Daniel Seemaier** — KaHIP, HeiStream
- **Darren Strash** — VieCut, KaMIS, KaHIP
- **Jesper Larsson Träff** — KaHIP
- **Renato F. Werneck** — KaMIS
- **Robert Williger** — KaMIS
- **Huashuo Zhang** — KaMIS
- **Bogdán Zaválnij** — KaMIS

---

<p align="center">
  <sub>MIT License</sub>
</p>
