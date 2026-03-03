# Dynamic Weighted Independent Set (DynWMIS)

**Original Repository:** [https://github.com/DynGraphLab/DynWMIS](https://github.com/DynGraphLab/DynWMIS)

---

## Overview

DynWMIS maintains a **weighted independent set** on a dynamic graph where edges can be inserted and deleted incrementally. Node weights are fixed at construction time. The goal is to maintain an independent set of high total weight at all times as the graph changes.

Seven algorithms are available, ranging from fast heuristics to higher-quality local search methods:

| Algorithm | Type | Description |
|-----------|------|-------------|
| `"simple"` | Heuristic | Simple greedy updates |
| `"one_fast"` | Heuristic | Fast single-step local search |
| `"greedy"` | Heuristic | Greedy weight-based updates |
| `"deg_greedy"` | Heuristic | Degree-aware greedy (default, best trade-off) |
| `"bfs"` | Heuristic | BFS-based neighborhood exploration |
| `"static"` | Exact | Recompute from scratch after each update |
| `"one_strong"` | Heuristic | Strong single-step local search |

---

## `DynWeightedMIS`

### Constructor

```python
DynWeightedMIS(
    num_nodes: int,
    node_weights: np.ndarray | list[int],
    algorithm: str = "deg_greedy",
    seed: int = 0,
    bfs_depth: int = 10,
    time_limit: float = 1000.0,
)
```

Or via the namespace:

```python
DynamicProblems.weighted_mis(
    num_nodes: int,
    node_weights: np.ndarray | list[int],
    algorithm: str = "deg_greedy",
    seed: int = 0,
    bfs_depth: int = 10,
    time_limit: float = 1000.0,
) -> DynWeightedMIS
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | `int` | *required* | Number of vertices. |
| `node_weights` | `array-like` | *required* | Node weight array (length `num_nodes`, int32). Fixed at construction. |
| `algorithm` | `str` | `"deg_greedy"` | Algorithm name (see table above). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `bfs_depth` | `int` | `10` | BFS depth for local algorithms (used by `"bfs"` and others). |
| `time_limit` | `float` | `1000.0` | Time limit in seconds for local solver. |

### Methods

| Method | Description |
|--------|-------------|
| `insert_edge(u, v)` | Insert an undirected edge (u, v). |
| `delete_edge(u, v)` | Delete an undirected edge (u, v). |
| `get_current_solution()` | Return the current independent set as a `DynWMISResult`. |

### Returns (`get_current_solution`)

**`DynWMISResult`**

| Field | Type | Description |
|-------|------|-------------|
| `weight` | `int` | Total weight of the independent set. |
| `vertices` | `np.ndarray` (bool) | Boolean array: `True` if vertex is in the independent set. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `algorithm` is not one of the valid choices. |

### Example

```python
from chszlablib import DynamicProblems
import numpy as np

# Create weighted dynamic MIS solver
weights = np.array([10, 20, 15, 30, 5], dtype=np.int32)
solver = DynamicProblems.weighted_mis(
    num_nodes=5,
    node_weights=weights,
    algorithm="deg_greedy",
)

# Build a graph
solver.insert_edge(0, 1)
solver.insert_edge(1, 2)
solver.insert_edge(2, 3)
solver.insert_edge(3, 4)

result = solver.get_current_solution()
print(f"Total MIS weight: {result.weight}")
print(f"In MIS: {[i for i, v in enumerate(result.vertices) if v]}")

# Dynamic update
solver.delete_edge(1, 2)
result = solver.get_current_solution()
print(f"Weight after deletion: {result.weight}")
```

---

## Performance Disclaimer

> This Python interface wraps the DynWMIS C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary -- especially for fine-grained per-edge updates. **For maximum performance with high update throughput, use the original C++ implementation directly from the [DynWMIS repository](https://github.com/DynGraphLab/DynWMIS).**

---

## References

- Jannick Borowitz, Christian Schulz, and Bernhard Schuster. "Dynamic Weighted Independent Sets." *arXiv preprint*, 2024.
