# Maximum Cut (fpt-max-cut)

**Original Repository:** [https://github.com/DataReductionMaxCut/fpt-max-cut](https://github.com/DataReductionMaxCut/fpt-max-cut)

---

## Overview

The **maximum cut problem** asks for a partition of the vertex set into two sets *S* and *V \ S* that **maximizes** the total weight of edges crossing the partition. This is the dual of the minimum cut problem and is NP-hard.

The solver applies **FPT (Fixed-Parameter Tractable) kernelization** rules, parameterized by the number of edges above the Edwards bound, to reduce the instance before solving. Two solving strategies are available:

| Method | Description |
|--------|-------------|
| `"heuristic"` | Kernelization + MQLib-based heuristic solver. Fast, suitable for large graphs. |
| `"exact"` | Kernelization + brute-force on the reduced kernel. Feasible when kernelization reduces the instance sufficiently. |

---

## `Decomposition.maxcut()`

### Signature

```python
Decomposition.maxcut(
    g: Graph,
    method: str = "heuristic",
    time_limit: float = 1.0,
) -> MaxCutResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, optionally weighted). |
| `method` | `str` | `"heuristic"` | Solver method: `"heuristic"` or `"exact"`. |
| `time_limit` | `float` | `1.0` | Time limit in seconds for the solver. |

### Returns

**`MaxCutResult`**

| Field | Type | Description |
|-------|------|-------------|
| `cut_value` | `int` | Weight of the maximum cut. |
| `partition` | `np.ndarray` (int32) | 0/1 array indicating which side of the cut each node belongs to. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `method` is not `"heuristic"` or `"exact"`. |
| `ValueError` | `time_limit < 0`. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("graph.graph")

# Heuristic solution
mc = Decomposition.maxcut(g, method="heuristic", time_limit=5.0)
print(f"Max-cut value: {mc.cut_value}")

# Exact solution (feasible for small/well-reducible instances)
mc_exact = Decomposition.maxcut(g, method="exact", time_limit=60.0)
print(f"Exact max-cut: {mc_exact.cut_value}")
```

---

## Performance Disclaimer

> This Python interface wraps the fpt-max-cut C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [fpt-max-cut repository](https://github.com/DataReductionMaxCut/fpt-max-cut).**

---

## References

```bibtex
@inproceedings{ferizovic2020engineering,
  author    = {Ferizovic, Damir and Hespe, Demian and Lamm, Sebastian and Mnich, Matthias and Schulz, Christian and Strash, Darren},
  title     = {Engineering Kernelization for Maximum Cut},
  booktitle = {Proceedings of the Symposium on Algorithm Engineering and Experiments (ALENEX)},
  publisher = {{SIAM}},
  year      = {2020},
  doi       = {10.1137/1.9781611976007.3}
}
```
