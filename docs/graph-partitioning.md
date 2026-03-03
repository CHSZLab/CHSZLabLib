# Graph Partitioning (KaHIP)

**Original Repository:** [https://github.com/KaHIP/KaHIP](https://github.com/KaHIP/KaHIP)

---

## Overview

KaHIP (Karlsruhe High Quality Partitioning) is a family of graph partitioning programs that tackle the balanced graph partitioning problem: given an undirected graph *G = (V, E)* with optional node and edge weights, partition the node set into *k* disjoint blocks of roughly equal weight while minimizing the total weight of edges crossing block boundaries (the **edge cut**).

The balance constraint ensures each block's weight does not exceed `(1 + imbalance) * ceil(total_weight / k)`.

KaHIP uses a **multilevel approach** with coarsening, initial partitioning, and local search refinement. CHSZLabLib exposes three KaHIP routines:

| Method | Purpose |
|--------|---------|
| `Decomposition.partition()` | Balanced *k*-way graph partitioning |
| `Decomposition.node_separator()` | Balanced node separator computation |
| `Decomposition.node_ordering()` | Fill-reducing nested dissection ordering |

---

## `Decomposition.partition()`

Partition a graph into *k* balanced blocks minimizing the edge cut.

### Signature

```python
Decomposition.partition(
    g: Graph,
    num_parts: int = 2,
    mode: str = "eco",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
) -> PartitionResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, optionally weighted). Must be finalized. |
| `num_parts` | `int` | `2` | Number of blocks (must be >= 2). |
| `mode` | `str` | `"eco"` | Quality/speed trade-off (see table below). |
| `imbalance` | `float` | `0.03` | Allowed weight imbalance as a fraction (0.03 = 3%). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

#### Available Modes

| Mode | Speed | Quality | Best For |
|------|-------|---------|----------|
| `"fast"` | Fastest | Good | Large-scale exploration |
| `"eco"` | Balanced | Very good | Default choice |
| `"strong"` | Slowest | Best | Final production partitions |
| `"fastsocial"` | Fastest | Good | Social / power-law networks |
| `"ecosocial"` | Balanced | Very good | Social / power-law networks |
| `"strongsocial"` | Slowest | Best | Social / power-law networks |

### Returns

**`PartitionResult`**

| Field | Type | Description |
|-------|------|-------------|
| `edgecut` | `int` | Total weight of edges crossing block boundaries. |
| `assignment` | `np.ndarray` (int32) | Block ID for each node (0-indexed). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `num_parts < 2` or `imbalance < 0`. |
| `InvalidModeError` | `mode` is not one of the valid choices. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("mesh.graph")
result = Decomposition.partition(g, num_parts=8, mode="strong", imbalance=0.01)
print(f"Edge cut: {result.edgecut}")
print(f"Block assignments: {result.assignment}")
```

---

## `Decomposition.node_separator()`

Compute a balanced node separator that splits the graph into two components with no edges between them.

### Signature

```python
Decomposition.node_separator(
    g: Graph,
    num_parts: int = 2,
    mode: str = "eco",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
) -> SeparatorResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, optionally weighted). |
| `num_parts` | `int` | `2` | Number of blocks (must be >= 2). |
| `mode` | `str` | `"eco"` | Quality/speed trade-off (same modes as `partition()`). |
| `imbalance` | `float` | `0.03` | Allowed weight imbalance (0.03 = 3%). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

### Returns

**`SeparatorResult`**

| Field | Type | Description |
|-------|------|-------------|
| `num_separator_vertices` | `int` | Number of nodes in the separator. |
| `separator` | `np.ndarray` (int32) | Array where each entry is 0 (block A), 1 (block B), or 2 (separator). |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("mesh.graph")
result = Decomposition.node_separator(g, mode="strong")
print(f"Separator size: {result.num_separator_vertices}")
separator_nodes = [i for i, v in enumerate(result.separator) if v == 2]
```

---

## `Decomposition.node_ordering()`

Compute a fill-reducing nested dissection ordering for sparse matrix factorization.

### Signature

```python
Decomposition.node_ordering(
    g: Graph,
    mode: str = "eco",
    seed: int = 0,
    suppress_output: bool = True,
) -> OrderingResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph representing the sparsity pattern. |
| `mode` | `str` | `"eco"` | Quality/speed trade-off (same modes as `partition()`). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

### Returns

**`OrderingResult`**

| Field | Type | Description |
|-------|------|-------------|
| `ordering` | `np.ndarray` (int32) | Permutation array. Node *i* should be placed at position `ordering[i]`. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("sparse_matrix.graph")
result = Decomposition.node_ordering(g, mode="strong")
perm = result.ordering
```

---

## Performance Disclaimer

> This Python interface wraps the KaHIP C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances or in production HPC pipelines, use the original C++ implementation directly from the [KaHIP repository](https://github.com/KaHIP/KaHIP).**

---

## References

```bibtex
@inproceedings{sandersschulz2013,
  author    = {Sanders, Peter and Schulz, Christian},
  title     = {{Think Locally, Act Globally: Highly Balanced Graph Partitioning}},
  booktitle = {Proceedings of the 12th International Symposium on Experimental Algorithms (SEA'13)},
  series    = {LNCS},
  publisher = {Springer},
  year      = {2013},
  volume    = {7933},
  pages     = {164--175}
}

@inproceedings{meyerhenkesandersschulz2017,
  author  = {Meyerhenke, Henning and Sanders, Peter and Schulz, Christian},
  title   = {{Parallel Graph Partitioning for Complex Networks}},
  journal = {IEEE Transactions on Parallel and Distributed Systems (TPDS)},
  volume  = {28},
  number  = {9},
  pages   = {2625--2638},
  year    = {2017}
}
```
