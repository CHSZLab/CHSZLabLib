# Correlation Clustering (SCC)

**Original Repository:** [https://github.com/ScalableCorrelationClustering/ScalableCorrelationClustering](https://github.com/ScalableCorrelationClustering/ScalableCorrelationClustering)

---

## Overview

**Correlation clustering** operates on graphs with **signed edge weights**: positive weights indicate similarity, negative weights indicate dissimilarity. The goal is to partition the vertex set into an arbitrary number of clusters that minimizes the total **disagreements** -- positive edges that cross cluster boundaries plus negative edges that remain within the same cluster.

Unlike standard clustering, the number of clusters is not fixed a priori but determined automatically by the optimization. CHSZLabLib provides two SCC variants:

| Method | Description |
|--------|-------------|
| `correlation_clustering()` | Multilevel label propagation (fast) |
| `evolutionary_correlation_clustering()` | Memetic evolutionary algorithm (higher quality, longer runtime) |

---

## `Decomposition.correlation_clustering()`

### Signature

```python
Decomposition.correlation_clustering(
    g: Graph,
    seed: int = 0,
    time_limit: float = 0,
) -> CorrelationClusteringResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph with **signed** edge weights. Positive = similarity, negative = dissimilarity. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `time_limit` | `float` | `0` | Time limit in seconds (0 = no limit). |

### Returns

**`CorrelationClusteringResult`**

| Field | Type | Description |
|-------|------|-------------|
| `edge_cut` | `int` | Number of disagreements. |
| `num_clusters` | `int` | Number of clusters found. |
| `assignment` | `np.ndarray` (int32) | Cluster ID for each node (0-indexed). |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph(num_nodes=4)
g.add_edge(0, 1, weight=1)    # similar
g.add_edge(1, 2, weight=-1)   # dissimilar
g.add_edge(2, 3, weight=1)    # similar
g.add_edge(0, 3, weight=-1)   # dissimilar
g.finalize()

result = Decomposition.correlation_clustering(g, seed=42)
print(f"Clusters: {result.num_clusters}, disagreements: {result.edge_cut}")
```

---

## `Decomposition.evolutionary_correlation_clustering()`

A higher-quality variant that uses a **population-based memetic evolutionary algorithm** with recombination and multilevel local search.

### Signature

```python
Decomposition.evolutionary_correlation_clustering(
    g: Graph,
    seed: int = 0,
    time_limit: float = 5.0,
) -> CorrelationClusteringResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph with signed edge weights. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `time_limit` | `float` | `5.0` | Time budget in seconds for the evolutionary search. |

### Returns

**`CorrelationClusteringResult`** (same fields as above).

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("signed_graph.graph")

# Fast version
fast = Decomposition.correlation_clustering(g, seed=42)
print(f"Fast: {fast.num_clusters} clusters, {fast.edge_cut} disagreements")

# Evolutionary refinement
evo = Decomposition.evolutionary_correlation_clustering(g, time_limit=30.0, seed=42)
print(f"Evo:  {evo.num_clusters} clusters, {evo.edge_cut} disagreements")
```

---

## Performance Disclaimer

> This Python interface wraps the SCC C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [SCC repository](https://github.com/ScalableCorrelationClustering/ScalableCorrelationClustering).**

---

## References

```bibtex
@misc{hausberger2024scalablemultilevelmemeticsigned,
  title         = {Scalable Multilevel and Memetic Signed Graph Clustering},
  author        = {Felix Hausberger and Marcelo Fonseca Faraj and Christian Schulz},
  year          = {2024},
  eprint        = {2208.13618},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  url           = {https://arxiv.org/abs/2208.13618}
}
```
