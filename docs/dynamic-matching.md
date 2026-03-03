# Dynamic Graph Matching (DynMatch)

**Original Repository:** [https://github.com/DynGraphLab/DynMatch](https://github.com/DynGraphLab/DynMatch)

---

## Overview

DynMatch maintains a **matching** on a dynamic graph where edges can be inserted and deleted incrementally. A matching is a set of edges such that no two edges share a vertex. The goal is to maintain a large matching at all times as the graph changes.

Seven algorithms are available, spanning exact and approximation approaches:

| Algorithm | Type | Description |
|-----------|------|-------------|
| `"blossom"` | Exact | Blossom algorithm with augmenting paths (default) |
| `"blossom_naive"` | Exact | Naive blossom variant |
| `"static_blossom"` | Exact | Recompute from scratch after each update |
| `"random_walk"` | Approximate | Random walk based matching |
| `"baswana_gupta_sen"` | Approximate | 2-approximate dynamic matching |
| `"neiman_solomon"` | Approximate | Neiman-Solomon dynamic matching |
| `"naive"` | Heuristic | Naive greedy matching |

---

## `DynMatching`

### Constructor

```python
DynMatching(
    num_nodes: int,
    algorithm: str = "blossom",
    seed: int = 0,
)
```

Or via the namespace:

```python
DynamicProblems.matching(
    num_nodes: int,
    algorithm: str = "blossom",
    seed: int = 0,
) -> DynMatching
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | `int` | *required* | Number of vertices. |
| `algorithm` | `str` | `"blossom"` | Algorithm name (see table above). |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Methods

| Method | Description |
|--------|-------------|
| `insert_edge(u, v)` | Insert an undirected edge (u, v). |
| `delete_edge(u, v)` | Delete an undirected edge (u, v). |
| `get_current_solution()` | Return the current matching as a `DynMatchingResult`. |

### Returns (`get_current_solution`)

**`DynMatchingResult`**

| Field | Type | Description |
|-------|------|-------------|
| `matching_size` | `int` | Number of matched edges. |
| `matching` | `np.ndarray` (int32) | Matching array: `matching[v]` = mate of vertex `v`, or `-1` if `v` is unmatched. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `algorithm` is not one of the valid choices. |

### Example

```python
from chszlablib import DynamicProblems

solver = DynamicProblems.matching(num_nodes=100, algorithm="blossom")

# Build a graph
solver.insert_edge(0, 1)
solver.insert_edge(2, 3)
solver.insert_edge(1, 2)

result = solver.get_current_solution()
print(f"Matching size: {result.matching_size}")

# Check who is matched to whom
for v in range(4):
    mate = result.matching[v]
    if mate >= 0:
        print(f"  Node {v} matched with node {mate}")

# Dynamic update
solver.delete_edge(0, 1)
result = solver.get_current_solution()
print(f"Matching size after deletion: {result.matching_size}")
```

---

## Performance Disclaimer

> This Python interface wraps the DynMatch C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary -- especially for fine-grained per-edge updates. **For maximum performance with high update throughput, use the original C++ implementation directly from the [DynMatch repository](https://github.com/DynGraphLab/DynMatch).**

---

## References

```bibtex
@inproceedings{DBLP:conf/esa/Henzinger0P020,
  author    = {Monika Henzinger and Shahbaz Khan and Richard Paul and Christian Schulz},
  title     = {Dynamic Matching Algorithms in Practice},
  booktitle = {28th Annual European Symposium on Algorithms, {ESA} 2020},
  pages     = {58:1--58:20},
  year      = {2020},
  url       = {https://doi.org/10.4230/LIPIcs.ESA.2020.58},
  doi       = {10.4230/LIPIcs.ESA.2020.58}
}
```
