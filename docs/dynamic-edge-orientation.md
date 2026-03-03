# Dynamic Edge Orientation (DynDeltaOrientation / DynDeltaApprox)

**Original Repositories:**
- [https://github.com/DynGraphLab/DynDeltaOrientation](https://github.com/DynGraphLab/DynDeltaOrientation)
- [https://github.com/DynGraphLab/DynDeltaApprox](https://github.com/DynGraphLab/DynDeltaApprox)

---

## Overview

These libraries maintain an **edge orientation** on a dynamic graph where edges can be inserted and deleted incrementally. The goal is to minimize the maximum out-degree at all times.

CHSZLabLib provides two solver classes with different algorithmic families:

| Class | Description | Algorithms |
|-------|-------------|------------|
| `DynEdgeOrientation` | Exact/heuristic dynamic orientation | 12 algorithms |
| `DynDeltaApproxOrientation` | Approximate dynamic orientation with bounded guarantees | 7 algorithms |

Both are accessible via the `DynamicProblems` namespace or directly.

---

## `DynEdgeOrientation`

Maintains an exact or heuristic edge orientation under insertions and deletions.

### Constructor

```python
DynEdgeOrientation(
    num_nodes: int,
    algorithm: str = "kflips",
    seed: int = 0,
)
```

Or via the namespace:

```python
DynamicProblems.edge_orientation(
    num_nodes: int,
    algorithm: str = "kflips",
    seed: int = 0,
) -> DynEdgeOrientation
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | `int` | *required* | Number of vertices. |
| `algorithm` | `str` | `"kflips"` | Algorithm name (see table below). |
| `seed` | `int` | `0` | Random seed for reproducibility. |

#### Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| `"bfs"` | BFS-based path search |
| `"naive_opt"` | Optimized naive reorientation |
| `"impro_opt"` | Improved optimized reorientation |
| `"kflips"` | k-flip local search (default, best overall) |
| `"rwalk"` | Random walk based |
| `"naive"` | Naive reorientation |
| `"brodal_fagerberg"` | Brodal-Fagerberg algorithm |
| `"max_descending"` | Maximum descending path |
| `"strong_opt"` | Strong optimization |
| `"strong_opt_dfs"` | Strong optimization with DFS |
| `"improved_opt"` | Improved optimization |
| `"improved_opt_dfs"` | Improved optimization with DFS |

### Methods

| Method | Description |
|--------|-------------|
| `insert_edge(u, v)` | Insert an undirected edge (u, v). |
| `delete_edge(u, v)` | Delete an undirected edge (u, v). |
| `get_current_solution()` | Return the current orientation as a `DynOrientationResult`. |

### Returns (`get_current_solution`)

**`DynOrientationResult`**

| Field | Type | Description |
|-------|------|-------------|
| `max_out_degree` | `int` | Maximum out-degree across all vertices. |
| `out_degrees` | `np.ndarray` (int32) | Out-degree array for all vertices. |

### Example

```python
from chszlablib import DynamicProblems

solver = DynamicProblems.edge_orientation(num_nodes=100, algorithm="kflips")

# Build graph incrementally
solver.insert_edge(0, 1)
solver.insert_edge(1, 2)
solver.insert_edge(2, 0)

result = solver.get_current_solution()
print(f"Max out-degree: {result.max_out_degree}")

# Dynamic update
solver.delete_edge(0, 1)
solver.insert_edge(0, 3)
result = solver.get_current_solution()
print(f"Max out-degree after update: {result.max_out_degree}")
```

---

## `DynDeltaApproxOrientation`

Maintains an approximate edge orientation with bounded maximum out-degree. Designed for scenarios where theoretical guarantees on the approximation factor are needed.

### Constructor

```python
DynDeltaApproxOrientation(
    num_nodes: int,
    num_edges_hint: int = 0,
    algorithm: str = "improved_bfs",
    lambda_param: float = 0.1,
    theta: int = 0,
    b: int = 1,
    bfs_depth: int = 20,
)
```

Or via the namespace:

```python
DynamicProblems.approx_edge_orientation(
    num_nodes: int,
    num_edges_hint: int = 0,
    algorithm: str = "improved_bfs",
    lambda_param: float = 0.1,
    theta: int = 0,
    b: int = 1,
    bfs_depth: int = 20,
) -> DynDeltaApproxOrientation
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | `int` | *required* | Number of vertices. |
| `num_edges_hint` | `int` | `0` | Hint for max number of edges (for memory pre-allocation in CCHHQRS variants). |
| `algorithm` | `str` | `"improved_bfs"` | Algorithm name (see table below). |
| `lambda_param` | `float` | `0.1` | Lambda parameter for CCHHQRS variants. |
| `theta` | `int` | `0` | Theta parameter for CCHHQRS variants. |
| `b` | `int` | `1` | Fractional edge parameter for CCHHQRS variants. |
| `bfs_depth` | `int` | `20` | BFS depth for BFS-based algorithms. |

#### Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| `"cchhqrs"` | CCHHQRS algorithm |
| `"limited_bfs"` | BFS with limited depth |
| `"strong_bfs"` | Strong BFS variant |
| `"improved_bfs"` | Improved BFS (default, best quality) |
| `"packed_cchhqrs"` | Packed CCHHQRS |
| `"packed_cchhqrs_list"` | Packed CCHHQRS with list |
| `"packed_cchhqrs_map"` | Packed CCHHQRS with map |

### Methods

| Method | Description |
|--------|-------------|
| `insert_edge(u, v)` | Insert an undirected edge (u, v). |
| `delete_edge(u, v)` | Delete an undirected edge (u, v). |
| `get_current_solution()` | Return the current maximum out-degree as an `int`. |

### Example

```python
from chszlablib import DynamicProblems

solver = DynamicProblems.approx_edge_orientation(
    num_nodes=1000,
    num_edges_hint=5000,
    algorithm="improved_bfs",
    bfs_depth=30,
)

solver.insert_edge(0, 1)
solver.insert_edge(1, 2)
solver.insert_edge(2, 0)

max_degree = solver.get_current_solution()
print(f"Max out-degree: {max_degree}")
```

---

## Performance Disclaimer

> These Python interfaces wrap the DynDeltaOrientation and DynDeltaApprox C++ libraries via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary -- especially for fine-grained per-edge updates. **For maximum performance with high update throughput, use the original C++ implementations directly from the [DynDeltaOrientation](https://github.com/DynGraphLab/DynDeltaOrientation) and [DynDeltaApprox](https://github.com/DynGraphLab/DynDeltaApprox) repositories.**

---

## References

- Monika Henzinger, Stefan Neumann, and Christian Schulz. "Dynamic Edge Orientation." *Proceedings of the 22nd Workshop on Algorithm Engineering and Experiments (ALENEX)*, 2020.
- Jingbang Chen, Li Chen, Rudy Huang, Xiaorui Sun, and Mikkel Thorup. "CCHHQRS: An Improved Algorithm for Dynamic Edge Orientation." *Proceedings of SODA*, 2024.
