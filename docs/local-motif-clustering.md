# Local Motif Clustering (HeidelbergMotifClustering)

**Original Repository:** [https://github.com/LocalClustering/HeidelbergMotifClustering](https://github.com/LocalClustering/HeidelbergMotifClustering)

---

## Overview

HeidelbergMotifClustering performs **local community detection** around a given seed node. Unlike global clustering algorithms that process the entire graph, this method explores only the neighborhood of the seed node via BFS and finds a cluster that minimizes the **triangle-motif conductance**.

Triangle-motif conductance measures the ratio of triangles cut by the cluster boundary to the minimum of triangles inside vs. outside the cluster. This higher-order metric captures structural cohesion more accurately than edge-based conductance, especially in social networks where triangles indicate strong community ties.

Two methods are available:

| Method | Description |
|--------|-------------|
| `"social"` | BFS extraction + triangle enumeration + MQI flow-based refinement. Faster. |
| `"lmchgp"` | BFS extraction + graph-partitioning-based approach. |

---

## `Decomposition.motif_cluster()`

### Signature

```python
Decomposition.motif_cluster(
    g: Graph,
    seed_node: int,
    method: str = "social",
    bfs_depths: list[int] | None = None,
    time_limit: int = 60,
    seed: int = 0,
) -> MotifClusterResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, unweighted). |
| `seed_node` | `int` | *required* | The node around which to find a local cluster (0-indexed). |
| `method` | `str` | `"social"` | Clustering method: `"social"` (flow-based) or `"lmchgp"` (partitioning-based). |
| `bfs_depths` | `list[int]` or `None` | `None` | BFS depths to explore around the seed node. Controls the size of the local subgraph. Defaults to `[10, 15, 20]`. |
| `time_limit` | `int` | `60` | Time limit in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`MotifClusterResult`**

| Field | Type | Description |
|-------|------|-------------|
| `cluster_nodes` | `np.ndarray` (int32) | Node IDs in the found cluster. |
| `motif_conductance` | `float` | Triangle-motif conductance of the cluster (lower is better). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `method` is not `"social"` or `"lmchgp"`. |
| `ValueError` | `seed_node` is out of range or `time_limit < 0`. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("social_network.graph")

# Find community around node 42
result = Decomposition.motif_cluster(g, seed_node=42, method="social")
print(f"Cluster size: {len(result.cluster_nodes)}")
print(f"Motif conductance: {result.motif_conductance:.4f}")
print(f"Cluster members: {result.cluster_nodes}")

# Custom BFS depths for finer control
result = Decomposition.motif_cluster(
    g, seed_node=42, bfs_depths=[5, 10, 15, 20, 25]
)
```

---

## Performance Disclaimer

> This Python interface wraps the HeidelbergMotifClustering C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [HeidelbergMotifClustering repository](https://github.com/LocalClustering/HeidelbergMotifClustering).**

---

## References

```bibtex
@inproceedings{DBLP:conf/alenex/ChhabraF023,
  author    = {Adil Chhabra and Marcelo Fonseca Faraj and Christian Schulz},
  title     = {Local Motif Clustering via (Hyper)Graph Partitioning},
  booktitle = {Proceedings of the 25th Symposium on Algorithm Engineering and Experiments,
               {ALENEX} 2023},
  pages     = {96--109},
  publisher = {{SIAM}},
  year      = {2023},
  doi       = {10.1137/1.9781611977561.ch9}
}

@inproceedings{DBLP:conf/esa/ChhabraF023,
  author    = {Adil Chhabra and Marcelo Fonseca Faraj and Christian Schulz},
  title     = {Faster Local Motif Clustering via Maximum Flows},
  booktitle = {31st Annual European Symposium on Algorithms, {ESA} 2023},
  series    = {LIPIcs},
  volume    = {274},
  pages     = {34:1--34:16},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2023},
  doi       = {10.4230/LIPIcs.ESA.2023.34}
}
```
