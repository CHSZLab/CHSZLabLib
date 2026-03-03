# Graph Clustering / Community Detection (VieClus)

**Original Repository:** [https://github.com/VieClus/VieClus](https://github.com/VieClus/VieClus)

---

## Overview

VieClus is a graph clustering algorithm that maximizes **Newman-Girvan modularity**. Given an undirected graph, it partitions the vertex set into an automatically determined number of clusters such that the density of edges within clusters is significantly higher than expected under a random graph model with the same degree sequence.

Modularity values range from -0.5 to 1.0, with higher values indicating stronger community structure. VieClus uses a **multilevel evolutionary algorithm** with population-based search, recombination operators, and local search refinement.

---

## `Decomposition.cluster()`

### Signature

```python
Decomposition.cluster(
    g: Graph,
    time_limit: float = 1.0,
    seed: int = 0,
    cluster_upperbound: int = 0,
    suppress_output: bool = True,
) -> ClusterResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, optionally weighted). |
| `time_limit` | `float` | `1.0` | Time budget in seconds for the evolutionary search. Longer budgets generally yield higher modularity. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `cluster_upperbound` | `int` | `0` | Maximum number of clusters allowed. 0 means no upper bound. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

### Returns

**`ClusterResult`**

| Field | Type | Description |
|-------|------|-------------|
| `modularity` | `float` | Achieved modularity value. |
| `num_clusters` | `int` | Number of clusters found. |
| `assignment` | `np.ndarray` (int32) | Cluster ID for each node (0-indexed). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `time_limit < 0`. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("social_network.graph")
result = Decomposition.cluster(g, time_limit=10.0)
print(f"Modularity: {result.modularity:.4f}")
print(f"Number of clusters: {result.num_clusters}")
print(f"Cluster assignments: {result.assignment}")

# Analyze cluster sizes
import numpy as np
unique, counts = np.unique(result.assignment, return_counts=True)
for cluster_id, size in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {size} nodes")
```

---

## Performance Disclaimer

> This Python interface wraps the VieClus C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [VieClus repository](https://github.com/VieClus/VieClus).**

---

## References

```bibtex
@inproceedings{BiedermannHSS18,
  author    = {Biedermann, Sonja and Henzinger, Monika and Schulz, Christian and Schuster, Bernhard},
  title     = {{Memetic Graph Clustering}},
  booktitle = {{Proceedings of the 17th International Symposium on Experimental Algorithms (SEA'18)}},
  series    = {{LIPIcs}},
  publisher = {Dagstuhl},
  note      = {Technical Report, arXiv:1802.07034},
  year      = {2018}
}
```
