# Streaming Graph Clustering (CluStRE)

**Original Repository:** [https://github.com/KaHIP/CluStRE](https://github.com/KaHIP/CluStRE)

---

## Overview

CluStRE (Clustering via Streaming and Refinement) is a streaming graph clustering algorithm that processes nodes sequentially, requiring only a fraction of the memory needed by in-memory methods. It uses a **Constant Potts Model (CPM)** objective with modularity-based cluster assignment and supports restreaming for improved quality.

Four quality modes are available:

| Mode | Description | Passes | Local Search |
|------|-------------|--------|--------------|
| `"light"` | Fastest, single pass | 1 | No |
| `"light_plus"` | Restreaming + local search | Multiple | Yes |
| `"evo"` | Evolutionary refinement | Multiple | Yes |
| `"strong"` | Best quality | Multiple | Yes |

CHSZLabLib provides two interfaces: a batch static method and a stateful streaming class.

---

## `Decomposition.stream_cluster()`

Cluster a graph using CluStRE when the full graph is available in memory.

### Signature

```python
Decomposition.stream_cluster(
    g: Graph,
    mode: str = "strong",
    seed: int = 0,
    num_streams_passes: int = 2,
    resolution_param: float = 0.5,
    max_num_clusters: int = -1,
    ls_time_limit: int = 600,
    ls_frac_time: float = 0.5,
    cut_off: float = 0.05,
    suppress_output: bool = True,
) -> StreamClusterResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, unweighted). Edge weights are ignored. |
| `mode` | `str` | `"strong"` | Clustering mode: `"light"`, `"light_plus"`, `"evo"`, or `"strong"`. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `num_streams_passes` | `int` | `2` | Number of streaming passes. More passes improve quality at the cost of runtime. |
| `resolution_param` | `float` | `0.5` | Resolution parameter for modularity. Higher values produce more (smaller) clusters. |
| `max_num_clusters` | `int` | `-1` | Maximum number of clusters. -1 means unlimited. |
| `ls_time_limit` | `int` | `600` | Local search time limit in seconds. |
| `ls_frac_time` | `float` | `0.5` | Fraction of total time allowed for local search. |
| `cut_off` | `float` | `0.05` | Convergence cut-off for local search. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

### Returns

**`StreamClusterResult`**

| Field | Type | Description |
|-------|------|-------------|
| `modularity` | `float` | Estimated modularity score. |
| `num_clusters` | `int` | Number of clusters found. |
| `assignment` | `np.ndarray` (int32) | Cluster ID for each node (0-indexed). |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_edge_list([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)])
result = Decomposition.stream_cluster(g, mode="strong")
print(f"{result.num_clusters} clusters, modularity={result.modularity:.4f}")
```

---

## `CluStReClusterer` (Stateful Streaming Class)

For true streaming scenarios where nodes arrive one at a time.

### Constructor

```python
CluStReClusterer(
    mode: str = "strong",
    seed: int = 0,
    num_streams_passes: int = 2,
    resolution_param: float = 0.5,
    max_num_clusters: int = -1,
    ls_time_limit: int = 600,
    ls_frac_time: float = 0.5,
    cut_off: float = 0.05,
    suppress_output: bool = True,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `new_node(node, neighbors)` | Add a node with its neighbor list. Non-contiguous IDs are supported. |
| `cluster()` | Run CluStRE on all accumulated nodes and return a `StreamClusterResult`. |
| `reset()` | Clear all added nodes while retaining configuration. |

### Example

```python
from chszlablib import CluStReClusterer

cs = CluStReClusterer(mode="strong")
cs.new_node(0, [1, 2])
cs.new_node(1, [0, 3])
cs.new_node(2, [0])
cs.new_node(3, [1])
result = cs.cluster()
print(f"{result.num_clusters} clusters, modularity={result.modularity:.4f}")
```

---

## Performance Disclaimer

> This Python interface wraps the CluStRE C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale streaming instances, use the original C++ implementation directly from the [CluStRE repository](https://github.com/KaHIP/CluStRE).**

---

## References

```bibtex
@inproceedings{DBLP:conf/wea/ChhabraP025,
  author    = {Adil Chhabra and Shai Dorian Peretz and Christian Schulz},
  title     = {CluStRE: Streaming Graph Clustering with Multi-Stage Refinement},
  booktitle = {23rd International Symposium on Experimental Algorithms, {SEA} 2025},
  series    = {LIPIcs},
  volume    = {338},
  pages     = {11:1--11:20},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2025},
  doi       = {10.4230/LIPIcs.SEA.2025.11}
}
```
