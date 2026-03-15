# Streaming Hypergraph Partitioning (FREIGHT)

**Original Repository:** [https://github.com/KaHIP/FREIGHT](https://github.com/KaHIP/FREIGHT)

---

## Overview

FREIGHT is a streaming algorithm for hypergraph partitioning based on the Fennel algorithm. It processes nodes in a single pass with O(k + |nets|) memory. Four algorithm variants are available:

| Algorithm | Identifier | Description |
|-----------|------------|-------------|
| Fennel Approx Sqrt | `"fennel_approx_sqrt"` | Default; fast square-root approximation of Fennel objective |
| Fennel | `"fennel"` | Full Fennel objective with exact power computation |
| LDG | `"ldg"` | Linear Deterministic Greedy |
| Hashing | `"hashing"` | Random hash-based assignment (fastest, lowest quality) |

Two objective functions: `"cut_net"` (default) and `"connectivity"`.

CHSZLabLib provides two interfaces: a batch static method and a true streaming class.

---

## `Decomposition.stream_hypergraph_partition()`

Partition a hypergraph using FREIGHT when the full hypergraph is available in memory.

### Signature

```python
Decomposition.stream_hypergraph_partition(
    hg: HyperGraph,
    k: int = 2,
    imbalance: float = 3.0,
    algorithm: str = "fennel_approx_sqrt",
    objective: str = "cut_net",
    seed: int = 0,
    num_streams_passes: int = 1,
    hierarchical: bool = False,
    rec_bisection_base: int = 2,
    suppress_output: bool = True,
) -> StreamHypergraphPartitionResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hg` | `HyperGraph` | *required* | Input hypergraph. |
| `k` | `int` | `2` | Number of partition blocks (must be >= 2). |
| `imbalance` | `float` | `3.0` | Allowed imbalance in percent (3.0 = 3%). |
| `algorithm` | `str` | `"fennel_approx_sqrt"` | Partitioning algorithm: `"fennel_approx_sqrt"`, `"fennel"`, `"ldg"`, or `"hashing"`. |
| `objective` | `str` | `"cut_net"` | Objective function: `"cut_net"` or `"connectivity"`. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `num_streams_passes` | `int` | `1` | Number of streaming passes (restreaming for improved quality). |
| `hierarchical` | `bool` | `False` | Enable hierarchical recursive bisection. |
| `rec_bisection_base` | `int` | `2` | Base for recursive bisection tree. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

### Returns

**`StreamHypergraphPartitionResult`**

| Field | Type | Description |
|-------|------|-------------|
| `assignment` | `np.ndarray` (int32) | Partition block ID for each node. |

### Example

```python
from chszlablib import HyperGraph, Decomposition

hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4], [4, 5, 0]], num_nodes=6)
result = Decomposition.stream_hypergraph_partition(hg, k=2, algorithm="fennel")
print(f"Assignment: {result.assignment}")
```

---

## `FreightPartitioner` (True Streaming Class)

For true streaming scenarios where nodes and their hypernets arrive one at a time. Each `assign_node()` call immediately returns the partition block ID. Memory: O(k + num_nets).

### Constructor

```python
FreightPartitioner(
    num_nodes: int,
    num_nets: int,
    k: int = 2,
    imbalance: float = 3.0,
    algorithm: str = "fennel_approx_sqrt",
    objective: str = "cut_net",
    seed: int = 0,
    hierarchical: bool = False,
    rec_bisection_base: int = 2,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `assign_node(node_id, nets, net_weights=None, node_weight=1)` | Assign a node given its incident nets (each net is a list of vertex IDs). Returns the partition block ID. |
| `get_assignment()` | Return a `StreamHypergraphPartitionResult` with the full assignment array. |

### Example

```python
from chszlablib import FreightPartitioner

fp = FreightPartitioner(num_nodes=6, num_nets=3, k=2)
fp.assign_node(0, nets=[[0, 1, 2], [0, 4, 5]])
fp.assign_node(1, nets=[[0, 1, 2]])
fp.assign_node(2, nets=[[0, 1, 2], [2, 3, 4]])
fp.assign_node(3, nets=[[2, 3, 4]])
fp.assign_node(4, nets=[[2, 3, 4], [0, 4, 5]])
fp.assign_node(5, nets=[[0, 4, 5]])
result = fp.get_assignment()
print(f"Assignment: {result.assignment}")
```

> **Note:** Nets are identified by their sorted vertex sets. The same net presented with vertices in different orders (e.g., `[0, 2, 1]` vs `[0, 1, 2]`) is automatically recognized as the same net.

---

## Performance Disclaimer

> This Python interface wraps the FREIGHT C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale streaming instances, use the original C++ implementation directly from the [FREIGHT repository](https://github.com/KaHIP/FREIGHT).**

---

## References

```bibtex
@InProceedings{FREIGHT2023,
  author    = {Kamal Eyubov and Marcelo Fonseca Faraj and Christian Schulz},
  title     = {{FREIGHT}: Fast Streaming Hypergraph Partitioning},
  booktitle = {21st International Symposium on Experimental Algorithms ({SEA})},
  series    = {LIPIcs},
  volume    = {265},
  pages     = {15:1--15:16},
  publisher = {Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  year      = {2023},
  doi       = {10.4230/LIPIcs.SEA.2023.15}
}
```
