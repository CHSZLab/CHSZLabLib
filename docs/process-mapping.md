# Hierarchical Process Mapping (SharedMap)

**Original Repository:** [https://github.com/HenningWoydt/SharedMap](https://github.com/HenningWoydt/SharedMap)

---

## Overview

SharedMap maps graph vertices (representing processes) to processing elements in a **hierarchical machine topology**, minimizing total communication cost. Given a communication graph where edge weights represent communication volume and a machine hierarchy description (e.g., 4 nodes x 8 cores), SharedMap assigns each process to a processing element such that frequently communicating processes are placed close together in the hierarchy.

Internally, SharedMap uses KaHIP for serial partitioning and Mt-KaHyPar for parallel partitioning, with configurable quality/speed trade-offs.

---

## `Decomposition.process_map()`

### Signature

```python
Decomposition.process_map(
    g: Graph,
    hierarchy: Sequence[int],
    distance: Sequence[int],
    *,
    mode: str | None = "eco",
    strategy: str | None = None,
    parallel_algorithm: str | None = None,
    serial_algorithm: str | None = None,
    imbalance: float = 0.03,
    threads: int = 1,
    seed: int = 0,
    verbose: bool = False,
) -> ProcessMappingResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Communication graph (undirected, weighted edges = communication volume). |
| `hierarchy` | `Sequence[int]` | *required* | Machine hierarchy levels, e.g., `[4, 8]` = 4 nodes x 8 cores = 32 PEs total. |
| `distance` | `Sequence[int]` | *required* | Communication cost per hierarchy level. Must have the same length as `hierarchy`. |
| `mode` | `str` or `None` | `"eco"` | Preset that selects strategy + algorithms (see table below). Set to `None` for full manual control. |
| `strategy` | `str` or `None` | `None` | Thread distribution strategy. Overrides mode preset. |
| `parallel_algorithm` | `str` or `None` | `None` | Parallel partitioning algorithm. Overrides mode preset. |
| `serial_algorithm` | `str` or `None` | `None` | Serial partitioning algorithm. Overrides mode preset. |
| `imbalance` | `float` | `0.03` | Allowed weight imbalance (0.03 = 3%). |
| `threads` | `int` | `1` | Number of threads for parallel partitioning. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `verbose` | `bool` | `False` | Print statistics from the C++ algorithm. |

#### Mode Presets

| Mode | Strategy | Parallel Algorithm | Serial Algorithm |
|------|----------|--------------------|------------------|
| `"fast"` | `nb_layer` | `mtkahypar_default` | `kaffpa_fast` |
| `"eco"` | `nb_layer` | `mtkahypar_default` | `kaffpa_eco` |
| `"strong"` | `nb_layer` | `mtkahypar_quality` | `kaffpa_strong` |

#### Available Strategies

| Strategy | Description |
|----------|-------------|
| `"naive"` | Simple top-down thread assignment |
| `"layer"` | Layer-by-layer distribution |
| `"queue"` | Queue-based distribution |
| `"nb_layer"` | Neighbor-aware layer distribution (default, best quality) |

#### Available Algorithms

**Parallel algorithms:** `"mtkahypar_default"`, `"mtkahypar_quality"`, `"mtkahypar_highest_quality"`

**Serial algorithms:** `"kaffpa_fast"`, `"kaffpa_eco"`, `"kaffpa_strong"`

### Returns

**`ProcessMappingResult`**

| Field | Type | Description |
|-------|------|-------------|
| `comm_cost` | `int` | Total communication cost of the mapping. |
| `assignment` | `np.ndarray` (int32) | PE assignment for each vertex (0-indexed). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | Invalid `mode`, `strategy`, or algorithm string. |
| `ValueError` | `hierarchy` and `distance` have different lengths, are empty, or contain invalid values. Also when `mode=None` but not all three algorithm parameters are specified. |

### Example

```python
from chszlablib import Graph, Decomposition

# Communication graph
g = Graph.from_edge_list([(0,1,10), (1,2,20), (2,3,10), (3,0,20)])

# Map to 2 nodes x 2 cores, with intra-node cost=1, inter-node cost=10
result = Decomposition.process_map(
    g,
    hierarchy=[2, 2],
    distance=[1, 10],
    mode="fast",
    threads=4,
)
print(f"Communication cost: {result.comm_cost}")
print(f"PE assignment: {result.assignment}")

# Manual configuration
result = Decomposition.process_map(
    g,
    hierarchy=[4, 8],
    distance=[1, 10],
    mode=None,
    strategy="nb_layer",
    parallel_algorithm="mtkahypar_quality",
    serial_algorithm="kaffpa_strong",
    threads=8,
)
```

---

## Performance Disclaimer

> This Python interface wraps the SharedMap C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [SharedMap repository](https://github.com/HenningWoydt/SharedMap).**

---

## References

- Henning Woydt and Christian Schulz. "SharedMap: Shared-Memory Process Mapping." *arXiv preprint*, 2024.
