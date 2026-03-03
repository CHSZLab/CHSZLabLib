# Streaming Graph Partitioning (HeiStream)

**Original Repository:** [https://github.com/KaHIP/HeiStream](https://github.com/KaHIP/HeiStream)

---

## Overview

HeiStream is a streaming graph partitioning algorithm designed for graphs that are too large to fit entirely in memory. Nodes and their adjacencies are presented sequentially, and each node is assigned to a block upon arrival (or after a bounded buffer delay).

HeiStream requires only **O(n + B)** memory, where *B* is the buffer size, compared to **O(n + m)** for full in-memory methods. It supports three operational modes:

| Mode | Description |
|------|-------------|
| **Fennel** | Direct one-pass assignment (`max_buffer_size` = 0 or 1) |
| **BuffCut** | Buffered assignment with local optimization (`max_buffer_size` > 1) |
| **Restreaming** | Multiple passes over the stream for improved quality (`num_streams_passes` > 1) |

CHSZLabLib provides two interfaces: a batch static method and a stateful streaming class.

---

## `Decomposition.stream_partition()`

Partition a graph using HeiStream when the full graph is available in memory.

### Signature

```python
Decomposition.stream_partition(
    g: Graph,
    k: int = 2,
    imbalance: float = 3.0,
    seed: int = 0,
    max_buffer_size: int = 0,
    batch_size: int = 0,
    num_streams_passes: int = 1,
    run_parallel: bool = False,
    suppress_output: bool = True,
) -> StreamPartitionResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, unweighted). Edge weights are ignored. |
| `k` | `int` | `2` | Number of partitions (must be >= 2). |
| `imbalance` | `float` | `3.0` | Allowed imbalance in percent (3.0 = 3%). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `max_buffer_size` | `int` | `0` | Buffer size for BuffCut mode. 0 or 1 = direct Fennel (no buffer). Larger values enable priority-buffer mode. |
| `batch_size` | `int` | `0` | MLP batch size for model-based partitioning within the buffer. 0 = HeiStream's default. |
| `num_streams_passes` | `int` | `1` | Number of streaming passes. More passes improve quality at the cost of runtime. |
| `run_parallel` | `bool` | `False` | Use the parallel 3-thread pipeline (I/O, PQ, partition). |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |

### Returns

**`StreamPartitionResult`**

| Field | Type | Description |
|-------|------|-------------|
| `assignment` | `np.ndarray` (int32) | Partition ID for each node (0-indexed). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `k < 2` or `imbalance < 0`. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("large_graph.graph")
result = Decomposition.stream_partition(g, k=8, imbalance=3.0, num_streams_passes=2)
print(f"Partition assignment: {result.assignment}")
```

---

## `HeiStreamPartitioner` (Stateful Streaming Class)

For true streaming scenarios where nodes arrive one at a time (e.g., from a network stream or database cursor).

### Constructor

```python
HeiStreamPartitioner(
    k: int = 2,
    imbalance: float = 3.0,
    seed: int = 0,
    max_buffer_size: int = 0,
    batch_size: int = 0,
    num_streams_passes: int = 1,
    run_parallel: bool = False,
    suppress_output: bool = True,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `new_node(node, neighbors)` | Add a node with its neighbor list to the stream. Nodes can be added in any order with non-contiguous IDs. |
| `partition()` | Run HeiStream on all accumulated nodes and return a `StreamPartitionResult`. |
| `reset()` | Clear all added nodes while retaining configuration. |

### Example

```python
from chszlablib import HeiStreamPartitioner

hs = HeiStreamPartitioner(k=4, imbalance=3.0, max_buffer_size=1000)
hs.new_node(0, [1, 2])
hs.new_node(1, [0, 3])
hs.new_node(2, [0])
hs.new_node(3, [1])
result = hs.partition()
print(result.assignment)  # array of partition IDs
```

---

## Performance Disclaimer

> This Python interface wraps the HeiStream C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale streaming instances, use the original C++ implementation directly from the [HeiStream repository](https://github.com/KaHIP/HeiStream).**

---

## References

- Marcelo Fonseca Faraj, Alexander van der Grinten, Henning Meyerhenke, Jesper Larsson Tr\"aff, and Christian Schulz. "Buffered Streaming Graph Partitioning." *ACM Journal of Experimental Algorithmics*, 2022.
