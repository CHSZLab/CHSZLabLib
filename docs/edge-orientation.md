# Static Edge Orientation (HeiOrient)

**Original Repository:** [https://github.com/HeiOrient/HeiOrient](https://github.com/HeiOrient/HeiOrient)

---

## Overview

HeiOrient orients the edges of an undirected graph to minimize the **maximum out-degree** of any vertex. The optimal maximum out-degree equals the **arboricity** of the graph, defined as:

```
arboricity(G) = max over all subgraphs H of ceil(|E(H)| / (|V(H)| - 1))
```

Low out-degree orientations enable:

- **Space-efficient adjacency queries** (each vertex stores only its out-neighbors)
- **Fast triangle enumeration** (enumerate triangles in O(m * arboricity) time)
- **Compact graph representations** for streaming and distributed settings

Three algorithms are available:

| Algorithm | Guarantee | Speed | Description |
|-----------|-----------|-------|-------------|
| `"two_approx"` | 2-approximation | Fast | Greedy orientation |
| `"dfs"` | Heuristic | Medium | DFS-based local search improvement |
| `"combined"` | Best quality | Slower | Combines greedy + DFS refinement |

---

## `Orientation.orient_edges()`

### Signature

```python
Orientation.orient_edges(
    g: Graph,
    algorithm: str = "combined",
    seed: int = 0,
    eager_size: int = 100,
) -> EdgeOrientationResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input undirected graph. |
| `algorithm` | `str` | `"combined"` | Algorithm: `"two_approx"`, `"dfs"`, or `"combined"`. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `eager_size` | `int` | `100` | Maximum path length for the eager path search heuristic (used by `"combined"`). |

### Returns

**`EdgeOrientationResult`**

| Field | Type | Description |
|-------|------|-------------|
| `max_out_degree` | `int` | Maximum out-degree achieved across all vertices. |
| `out_degrees` | `np.ndarray` (int32) | Out-degree of each vertex. |
| `edge_heads` | `np.ndarray` (int32) | Head (target) of each oriented edge (same length as `g.adjncy`). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `algorithm` is not one of the valid choices. |

### Example

```python
from chszlablib import Graph, Orientation

g = Graph.from_metis("graph.graph")

# Best quality orientation
result = Orientation.orient_edges(g, algorithm="combined")
print(f"Max out-degree: {result.max_out_degree}")
print(f"Out-degrees: {result.out_degrees}")

# Fast 2-approximation
fast = Orientation.orient_edges(g, algorithm="two_approx")
print(f"Fast max out-degree: {fast.max_out_degree}")
```

---

## Performance Disclaimer

> This Python interface wraps the HeiOrient C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [HeiOrient repository](https://github.com/HeiOrient/HeiOrient).**

---

## References

```bibtex
@inproceedings{DBLP:conf/esa/Reinstadtler0U24,
  author    = {Henrik Reinst{\"{a}}dtler and Christian Schulz and Bora U{\c{c}}ar},
  title     = {Engineering Edge Orientation Algorithms},
  booktitle = {32nd Annual European Symposium on Algorithms, {ESA} 2024},
  series    = {LIPIcs},
  volume    = {308},
  pages     = {97:1--97:18},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2024},
  doi       = {10.4230/LIPIcs.ESA.2024.97}
}
```
