# Exact Hypergraph Minimum Cut (HeiCut)

**Original Repository:** [https://github.com/heicut/heicut](https://github.com/heicut/heicut)

---

## Overview

HeiCut computes an **exact minimum cut** on hypergraphs: given a hypergraph *H = (V, E)* with optional vertex and hyperedge weights, find a bipartition of the vertex set that minimizes the total weight of hyperedges crossing the cut.

Four algorithmic strategies are available:

| Algorithm | Description | Notes |
|-----------|-------------|-------|
| `"kernelizer"` | Kernelization reductions + base solver | Fastest in practice |
| `"ilp"` | Integer linear programming | Requires `gurobipy` |
| `"submodular"` | Submodular function minimization | No external dependencies |
| `"trimmer"` | k-trimmed certificates | Unweighted hypergraphs only |

---

## `Decomposition.hypergraph_mincut()`

### Signature

```python
Decomposition.hypergraph_mincut(
    hg: HyperGraph,
    algorithm: str = "kernelizer",
    *,
    base_solver: str = "submodular",
    ilp_timeout: float = 7200.0,
    ilp_mode: str = "bip",
    ordering_type: str = "tight",
    ordering_mode: str = "single",
    seed: int = 0,
    threads: int = 1,
    unweighted: bool = False,
) -> HypergraphMincutResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hg` | `HyperGraph` | *required* | Input hypergraph (must be finalized). |
| `algorithm` | `str` | `"kernelizer"` | Algorithm: `"kernelizer"`, `"ilp"`, `"submodular"`, or `"trimmer"`. |
| `base_solver` | `str` | `"submodular"` | Base solver for the kernelizer: `"submodular"` or `"ilp"`. Ignored by other algorithms. |
| `ilp_timeout` | `float` | `7200.0` | Time limit in seconds for ILP-based solving. |
| `ilp_mode` | `str` | `"bip"` | ILP formulation: `"bip"` (binary IP) or `"milp"` (mixed ILP). Only for `"ilp"` algorithm. |
| `ordering_type` | `str` | `"tight"` | Vertex ordering: `"ma"` (maximum adjacency), `"tight"`, or `"queyranne"`. |
| `ordering_mode` | `str` | `"single"` | Ordering pass mode: `"single"` or `"multi"`. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `threads` | `int` | `1` | Number of threads. |
| `unweighted` | `bool` | `False` | If `True`, treat all edge weights as 1. |

### Returns

**`HypergraphMincutResult`**

| Field | Type | Description |
|-------|------|-------------|
| `cut_value` | `int` | Minimum edge cut value. |
| `time` | `float` | Computation time in seconds. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `algorithm` is not one of the valid choices. |
| `ValueError` | `threads < 1` or hypergraph is empty. |
| `TypeError` | Input is not a `HyperGraph` object. |
| `RuntimeError` | ILP solving fails (e.g., `gurobipy` not installed). |

### Example

```python
from chszlablib import HyperGraph, Decomposition

hg = HyperGraph.from_edge_list([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

# Default: kernelizer with submodular base solver
result = Decomposition.hypergraph_mincut(hg, algorithm="kernelizer")
print(f"Min cut: {result.cut_value} (computed in {result.time:.2f}s)")

# Submodular function minimization
result = Decomposition.hypergraph_mincut(hg, algorithm="submodular")
print(f"Min cut: {result.cut_value}")
```

---

## Performance Disclaimer

> This Python interface wraps the HeiCut C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [HeiCut repository](https://github.com/heicut/heicut).**

---

## References

```bibtex
@article{DBLP:journals/corr/abs-2504-19842,
  author     = {Adil Chhabra and Christian Schulz and Bora U{\c{c}}ar and Loris Wilwert},
  title      = {Near-Optimal Minimum Cuts in Hypergraphs at Scale},
  journal    = {CoRR},
  volume     = {abs/2504.19842},
  year       = {2025},
  eprinttype = {arXiv},
  eprint     = {2504.19842},
  doi        = {10.48550/arXiv.2504.19842}
}
```
