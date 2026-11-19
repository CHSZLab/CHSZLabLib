# Hypergraph Maximum Independent Set (HyperMIS)

**Original Repository:** [https://github.com/KarlsruheMIS/HyperMIS](https://github.com/KarlsruheMIS/HyperMIS)

---

## Overview

HyperMIS computes a **maximum independent set on a hypergraph**. Given a hypergraph *H = (V, E)* where each hyperedge contains two or more vertices, find a maximum independent set *I* such that for every hyperedge *e* with |e| >= 2, at most one vertex from *e* is in *I*. This is "strong" independence: every hyperedge may contribute at most one vertex to the solution.

HyperMIS first applies its kernelization reductions (including unconfined vertex removal), then solves the remaining kernel exactly via integer linear programming with `gurobipy` (requires a valid Gurobi license). `is_optimal` reports whether the ILP proved optimality within the time limit.

---

## `IndependenceProblems.hypermis()`

### Signature

```python
IndependenceProblems.hypermis(
    hg: HyperGraph,
    time_limit: float = 60.0,
    seed: int = 0,
) -> HyperMISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hg` | `HyperGraph` | *required* | Input hypergraph. |
| `time_limit` | `float` | `60.0` | Wall-clock time budget in seconds for reductions plus ILP (the ILP gets the time left after reductions). |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`HyperMISResult`**

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of vertices in the independent set. |
| `weight` | `int` | Total node weight of selected vertices. |
| `vertices` | `np.ndarray` (int32) | Vertex IDs in the independent set. |
| `offset` | `int` | Number of vertices determined during the reduction phase. |
| `reduction_time` | `float` | Wall-clock seconds spent on reductions. |
| `is_optimal` | `bool` | `True` if the ILP proved optimality within the time limit. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `time_limit < 0`. |
| `ImportError` | `gurobipy` is not installed. |

### Checking ILP Availability

```python
from chszlablib import IndependenceProblems

if IndependenceProblems.HYPERMIS_ILP_AVAILABLE:
    print("Gurobi is available, hypermis can be used")
else:
    print("Install gurobipy to use hypermis")
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

result = IndependenceProblems.hypermis(hg, time_limit=120.0)
print(f"IS size: {result.size}, optimal: {result.is_optimal}")
print(f"Reductions fixed {result.offset} vertices in {result.reduction_time:.2f}s")
```

---

## Performance Disclaimer

> This Python interface wraps the HyperMIS C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [HyperMIS repository](https://github.com/KarlsruheMIS/HyperMIS).**

---

## References

```bibtex
@misc{grossmann2026hypermis,
  author = {Ernestine Gro{\ss}mann and Christian Schulz and Darren Strash and Antonie Wagner},
  title  = {Data Reductions for the Strong Maximum Independent Set Problem in Hypergraphs},
  year   = {2026},
  note   = {Technical Report}
}
```
