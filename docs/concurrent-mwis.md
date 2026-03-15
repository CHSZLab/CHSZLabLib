# Concurrent Maximum Weight Independent Set (CHILS)

**Original Repository:** [https://github.com/KarlsruheMIS/CHILS](https://github.com/KarlsruheMIS/CHILS)

---

## Overview

CHILS (Concurrent Heuristic Independent set Local Search) computes a **maximum weight independent set (MWIS)** using multiple concurrent local search threads that explore different regions of the solution space simultaneously. It is designed for large instances where exact methods are infeasible.

CHILS features:

- **GNN-accelerated reductions** for faster instance shrinking
- **Concurrent local searches** running in parallel on shared data structures
- **Scalability** to very large graphs through aggressive reductions

---

## `IndependenceProblems.chils()`

### Signature

```python
IndependenceProblems.chils(
    g: Graph,
    time_limit: float = 10.0,
    num_concurrent: int = 4,
    seed: int = 0,
) -> MWISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph. Node weights are **required** and define the objective. |
| `time_limit` | `float` | `10.0` | Wall-clock time budget in seconds. |
| `num_concurrent` | `int` | `4` | Number of concurrent local search threads. |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`MWISResult`**

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of vertices in the independent set. |
| `weight` | `int` | Total weight of selected vertices. |
| `vertices` | `np.ndarray` (int32) | Vertex IDs in the independent set. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `time_limit < 0` or `num_concurrent < 1`. |

### Example

```python
from chszlablib import Graph, IndependenceProblems
import numpy as np

# Create a weighted graph
g = Graph.from_metis("weighted_graph.graph")

# Use 8 concurrent threads for 60 seconds
result = IndependenceProblems.chils(g, time_limit=60.0, num_concurrent=8)
print(f"MWIS size: {result.size}")
print(f"MWIS weight: {result.weight}")
print(f"Selected vertices: {result.vertices}")
```

---

## Performance Disclaimer

> This Python interface wraps the CHILS C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [CHILS repository](https://github.com/KarlsruheMIS/CHILS).**

---

## References

```bibtex
@article{DBLP:journals/corr/abs-2412-14198,
  author     = {Ernestine Gro{\ss}mann and Kenneth Langedal and Christian Schulz},
  title      = {Accelerating Reductions Using Graph Neural Networks and a New Concurrent
                Local Search for the Maximum Weight Independent Set Problem},
  journal    = {CoRR},
  volume     = {abs/2412.14198},
  year       = {2024},
  eprinttype = {arXiv},
  eprint     = {2412.14198},
  doi        = {10.48550/arXiv.2412.14198}
}
```
