# Hypergraph Maximum Independent Set (HyperMIS)

**Original Repository:** [https://github.com/KarlsruheMIS/HyperMIS](https://github.com/KarlsruheMIS/HyperMIS)

---

## Overview

HyperMIS computes a **maximum independent set on a hypergraph**. Given a hypergraph *H = (V, E)* where each hyperedge contains two or more vertices, find a maximum independent set *I* such that for every hyperedge *e* with |e| >= 2, at most one vertex from *e* is in *I*. This is "strong" independence: every hyperedge may contribute at most one vertex to the solution.

Two solving strategies are available:

| Method | Description | Requirements |
|--------|-------------|--------------|
| `"heuristic"` | Kernelization reductions + greedy heuristic peeling | None |
| `"exact"` | Kernelization reductions + ILP on remaining kernel | `gurobipy` + valid Gurobi license |

The heuristic method is fast but not provably optimal. The exact method solves the remaining kernel after reductions via integer linear programming and certifies optimality when feasible.

---

## `IndependenceProblems.hypermis()`

### Signature

```python
IndependenceProblems.hypermis(
    hg: HyperGraph,
    method: str = "heuristic",
    time_limit: float = 60.0,
    seed: int = 0,
    strong_reductions: bool = True,
) -> HyperMISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hg` | `HyperGraph` | *required* | Input hypergraph. |
| `method` | `str` | `"heuristic"` | Solving strategy: `"heuristic"` or `"exact"`. |
| `time_limit` | `float` | `60.0` | Wall-clock time budget in seconds. Also used as the Gurobi time limit for `"exact"`. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `strong_reductions` | `bool` | `True` | Enable aggressive reduction rules (unconfined vertices, larger edge-size threshold). |

### Returns

**`HyperMISResult`**

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of vertices in the independent set. |
| `weight` | `int` | Total node weight of selected vertices. |
| `vertices` | `np.ndarray` (int32) | Vertex IDs in the independent set. |
| `offset` | `int` | Number of vertices determined during the reduction phase. |
| `reduction_time` | `float` | Wall-clock seconds spent on reductions. |
| `is_optimal` | `bool` | `True` if the ILP proved optimality (only possible with `"exact"`). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `method` is not `"heuristic"` or `"exact"`. |
| `ValueError` | `time_limit < 0`. |
| `ImportError` | `method="exact"` but `gurobipy` is not installed. |

### Checking ILP Availability

```python
from chszlablib import IndependenceProblems

if IndependenceProblems.HYPERMIS_ILP_AVAILABLE:
    print("Gurobi is available for exact solving")
else:
    print("Only heuristic method available (install gurobipy for exact)")
```

### Example

```python
from chszlablib import HyperGraph, IndependenceProblems

hg = HyperGraph.from_edge_list([
    [0, 1, 2],
    [1, 2, 3],
    [3, 4, 5],
    [4, 5, 6],
])

# Heuristic solution
result = IndependenceProblems.hypermis(hg, method="heuristic")
print(f"IS size: {result.size}, optimal: {result.is_optimal}")
print(f"Reductions fixed {result.offset} vertices in {result.reduction_time:.2f}s")

# Exact solution (requires gurobipy)
if IndependenceProblems.HYPERMIS_ILP_AVAILABLE:
    exact = IndependenceProblems.hypermis(hg, method="exact", time_limit=120.0)
    print(f"Exact IS size: {exact.size}, optimal: {exact.is_optimal}")
```

---

## Performance Disclaimer

> This Python interface wraps the HyperMIS C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [HyperMIS repository](https://github.com/KarlsruheMIS/HyperMIS).**

---

## References

- Freja Bj\"ork Christensen, Alexander Gro{\ss}mann, and Christian Schulz. "Finding Maximum Independent Sets in Hypergraphs." *arXiv preprint*, 2024.
