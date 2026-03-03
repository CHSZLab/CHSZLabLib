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

```bibtex
@inproceedings{DBLP:conf/acda/BorowitzG023,
  author    = {Jannick Borowitz and Ernestine Gro{\ss}mann and Christian Schulz},
  title     = {Engineering Fully Dynamic {\(\Delta\)}-Orientation Algorithms},
  booktitle = {{SIAM} Conference on Applied and Computational Discrete Algorithms,
               {ACDA} 2023},
  pages     = {25--37},
  publisher = {{SIAM}},
  year      = {2023},
  doi       = {10.1137/1.9781611977714.3}
}

@article{DBLP:journals/corr/abs-2407-12595,
  author     = {Ernestine Gro{\ss}mann and Henrik Reinst{\"{a}}dtler
                and Christian Schulz and Fabian Walliser},
  title      = {Engineering Fully Dynamic Exact {\(\Delta\)}-Orientation Algorithms},
  journal    = {CoRR},
  volume     = {abs/2407.12595},
  year       = {2024},
  eprinttype = {arXiv},
  eprint     = {2407.12595},
  doi        = {10.48550/arXiv.2407.12595}
}

@inproceedings{DBLP:conf/esa/GrossmannRR0HV25,
  author    = {Ernestine Gro{\ss}mann and Henrik Reinst{\"{a}}dtler and Eva Rotenberg
               and Christian Schulz and Ivor {van der Hoog} and Juliette Vlieghe},
  title     = {From Theory to Practice: Engineering Approximation Algorithms for
               Dynamic Orientation},
  booktitle = {33rd Annual European Symposium on Algorithms, {ESA} 2025},
  series    = {LIPIcs},
  volume    = {351},
  pages     = {65:1--65:18},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2025},
  doi       = {10.4230/LIPIcs.ESA.2025.65}
}
```
