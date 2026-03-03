# Evolutionary Graph Partitioning (KaFFPaE)

**Original Repository:** [https://github.com/KaHIP/KaHIP](https://github.com/KaHIP/KaHIP)

---

## Overview

KaFFPaE (KaHIP -- Fast Flow Partitioner Evolutionary) is an evolutionary/memetic algorithm for balanced graph partitioning. It solves the same problem as standard KaHIP partitioning -- minimize the edge cut subject to balance constraints -- but uses a **population-based evolutionary strategy** that maintains and evolves a pool of partitions through recombination and multilevel local search over a configurable time budget.

KaFFPaE supports **warm-starting** from an existing partition, making it suitable for iterative refinement workflows where an initial solution is available from a faster method.

---

## `Decomposition.evolutionary_partition()`

### Signature

```python
Decomposition.evolutionary_partition(
    g: Graph,
    num_parts: int,
    time_limit: int,
    mode: str = "strong",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
    initial_partition: np.ndarray | None = None,
) -> PartitionResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (undirected, optionally weighted). |
| `num_parts` | `int` | *required* | Number of blocks (must be >= 2). |
| `time_limit` | `int` | *required* | Time budget in seconds for the evolutionary search. Longer budgets yield better partitions. |
| `mode` | `str` | `"strong"` | Quality/speed trade-off (see table below). |
| `imbalance` | `float` | `0.03` | Allowed weight imbalance as a fraction (0.03 = 3%). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `suppress_output` | `bool` | `True` | Suppress C++ stdout/stderr output. |
| `initial_partition` | `np.ndarray` or `None` | `None` | Optional warm-start partition (one block ID per node, int32). If provided, the evolutionary search begins from this solution. |

#### Available Modes

| Mode | Speed | Quality | Notes |
|------|-------|---------|-------|
| `"fast"` | Fastest | Good | Quick initial exploration |
| `"eco"` | Balanced | Very good | Good default |
| `"strong"` | Slowest | Best | Maximum quality |
| `"fastsocial"` | Fastest | Good | Optimized for social/power-law graphs |
| `"ecosocial"` | Balanced | Very good | Social/power-law graphs |
| `"strongsocial"` | Slowest | Best | Social/power-law graphs |
| `"ultrafastsocial"` | Ultra-fast | Baseline | Fastest possible for social graphs |

### Returns

**`PartitionResult`**

| Field | Type | Description |
|-------|------|-------------|
| `edgecut` | `int` | Total weight of edges crossing block boundaries. |
| `assignment` | `np.ndarray` (int32) | Block ID for each node (0-indexed). |
| `balance` | `float` | Achieved balance ratio. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `num_parts < 2`, `time_limit < 0`, or `imbalance < 0`. |
| `InvalidModeError` | `mode` is not one of the valid choices. |

### Example

```python
from chszlablib import Graph, Decomposition

g = Graph.from_metis("large_mesh.graph")

# Start with a fast partition, then refine
initial = Decomposition.partition(g, num_parts=16, mode="eco")
refined = Decomposition.evolutionary_partition(
    g,
    num_parts=16,
    time_limit=60,
    mode="strong",
    initial_partition=initial.assignment,
)
print(f"Initial edge cut: {initial.edgecut}")
print(f"Refined edge cut: {refined.edgecut} (balance: {refined.balance:.4f})")
```

### Warm-Start Workflow

```python
# Two-phase approach: quick partition + evolutionary refinement
fast_result = Decomposition.partition(g, num_parts=8, mode="fast")
final_result = Decomposition.evolutionary_partition(
    g, num_parts=8, time_limit=120,
    initial_partition=fast_result.assignment,
)
```

---

## Performance Disclaimer

> This Python interface wraps the KaFFPaE C++ implementation via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances or in production HPC pipelines, use the original C++ implementation directly from the [KaHIP repository](https://github.com/KaHIP/KaHIP).**

---

## References

- Peter Sanders and Christian Schulz. "Distributed Evolutionary Graph Partitioning." *Proceedings of the 12th Workshop on Algorithm Engineering and Experiments (ALENEX)*, 2012.
