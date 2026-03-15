# Global Minimum Cut (VieCut)

**Original Repository:** [https://github.com/VieCut/VieCut](https://github.com/VieCut/VieCut)

---

## Overview

VieCut computes the **global minimum cut** of an undirected graph: a partition of the vertex set into two non-empty sets *S* and *V \ S* that minimizes the total weight of edges crossing the partition. The resulting cut value equals the **edge connectivity** of the graph.

Minimum cuts identify the most vulnerable bottleneck in a network. Applications include network reliability analysis, image segmentation, and connectivity certification.

VieCut provides three algorithms with different exactness/performance trade-offs:

| Algorithm | Description | Complexity |
|-----------|-------------|------------|
| `"inexact"` | Heuristic VieCut (shared-memory parallel, near-linear time) | Fastest, not guaranteed optimal |
| `"exact"` | Shared-memory parallel exact algorithm | Guaranteed optimal |
| `"cactus"` | Enumerates all minimum cuts via cactus representation | Most comprehensive |

---

## `Decomposition.mincut()`

### Signature

```python
Decomposition.mincut(
    g: Graph,
    algorithm: str = "inexact",
    seed: int = 0,
) -> MincutResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, optionally weighted). |
| `algorithm` | `str` | `"inexact"` | Algorithm to use (see table above). |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`MincutResult`**

| Field | Type | Description |
|-------|------|-------------|
| `cut_value` | `int` | Weight of the minimum cut. |
| `partition` | `np.ndarray` (int32) | 0/1 array indicating which side of the cut each node belongs to. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `algorithm` is not `"inexact"`, `"exact"`, or `"cactus"`. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("network.graph")

# Fast heuristic
mc = Decomposition.mincut(g, algorithm="inexact")
print(f"Approximate min-cut: {mc.cut_value}")

# Exact computation
mc_exact = Decomposition.mincut(g, algorithm="exact")
print(f"Exact min-cut: {mc_exact.cut_value}")

# Vertices on each side
side_a = [i for i, v in enumerate(mc_exact.partition) if v == 0]
side_b = [i for i, v in enumerate(mc_exact.partition) if v == 1]
```

---

## Performance Disclaimer

> This Python interface wraps the VieCut C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [VieCut repository](https://github.com/VieCut/VieCut).**

---

## References

```bibtex
@article{DBLP:journals/jea/HenzingerNSS18,
  author  = {Monika Henzinger and Alexander Noe and Christian Schulz and Darren Strash},
  title   = {Practical Minimum Cut Algorithms},
  journal = {{ACM} J. Exp. Algorithmics},
  volume  = {23},
  year    = {2018},
  doi     = {10.1145/3274662}
}

@inproceedings{DBLP:conf/ipps/HenzingerN019,
  author    = {Monika Henzinger and Alexander Noe and Christian Schulz},
  title     = {Shared-Memory Exact Minimum Cuts},
  booktitle = {2019 {IEEE} International Parallel and Distributed Processing Symposium,
               {IPDPS} 2019},
  pages     = {13--22},
  publisher = {{IEEE}},
  year      = {2019},
  doi       = {10.1109/IPDPS.2019.00013}
}

@inproceedings{DBLP:conf/esa/HenzingerN0S20,
  author    = {Monika Henzinger and Alexander Noe and Christian Schulz and Darren Strash},
  title     = {Finding All Global Minimum Cuts in Practice},
  booktitle = {28th Annual European Symposium on Algorithms, {ESA} 2020},
  series    = {LIPIcs},
  volume    = {173},
  pages     = {59:1--59:20},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2020},
  doi       = {10.4230/LIPIcs.ESA.2020.59}
}
```
