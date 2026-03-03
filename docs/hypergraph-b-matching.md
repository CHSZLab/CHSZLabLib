# Hypergraph B-Matching (HeiHGM)

**Original Repositories:**
- Static: [https://github.com/HeiHGM/Bmatching](https://github.com/HeiHGM/Bmatching)
- Streaming: [https://github.com/HeiHGM/Streaming](https://github.com/HeiHGM/Streaming)

---

## Overview

HeiHGM provides algorithms for the **hypergraph b-matching problem**: given a hypergraph *H = (V, E)* with edge weights *w* and per-vertex capacities *b*, find a set of edges *M* (the matching) that maximizes the total weight `sum(w(e) for e in M)` subject to each vertex *v* being incident to at most *b(v)* matched edges.

When all capacities are 1, this reduces to standard maximum weight matching. CHSZLabLib provides two interfaces:

| Interface | Description |
|-----------|-------------|
| `IndependenceProblems.bmatching()` | Static batch solver (full hypergraph in memory) |
| `StreamingBMatcher` | True streaming solver (edges arrive one at a time) |

---

## `IndependenceProblems.bmatching()`

### Signature

```python
IndependenceProblems.bmatching(
    hg: HyperGraph,
    algorithm: str = "greedy_weight_desc",
    seed: int = 0,
    ils_iterations: int = 15,
    ils_time_limit: float = 1800.0,
    ILP_time_limit: float = 1000.0,
) -> BMatchingResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hg` | `HyperGraph` | *required* | Input hypergraph. Edge weights define the objective. Vertex capacities are set via `hg.set_capacity()` (default 1). |
| `algorithm` | `str` | `"greedy_weight_desc"` | Algorithm name (see table below). |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `ils_iterations` | `int` | `15` | Max ILS iterations (only for `"ils"`). |
| `ils_time_limit` | `float` | `1800.0` | ILS time budget in seconds (only for `"ils"`). |
| `ILP_time_limit` | `float` | `1000.0` | ILP time limit in seconds (only for `"reductions"`). |

#### Available Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| `"greedy_random"` | Greedy | Random edge ordering |
| `"greedy_weight_desc"` | Greedy | Heaviest edges first (default) |
| `"greedy_weight_asc"` | Greedy | Lightest edges first |
| `"greedy_degree_asc"` | Greedy | Lowest-degree edges first |
| `"greedy_degree_desc"` | Greedy | Highest-degree edges first |
| `"greedy_weight_degree_ratio_desc"` | Greedy | Best weight/degree ratio first |
| `"greedy_weight_degree_ratio_asc"` | Greedy | Worst weight/degree ratio first |
| `"reductions"` | Exact | Reduction rules + ILP solver |
| `"ils"` | Metaheuristic | Iterated local search |

### Returns

**`BMatchingResult`**

| Field | Type | Description |
|-------|------|-------------|
| `matched_edges` | `np.ndarray` (int32) | Indices of matched edges. |
| `total_weight` | `float` | Total weight of matched edges. |
| `num_matched` | `int` | Number of matched edges. |
| `is_optimal` | `bool` | Whether the ILP solver proved optimality (only for `"reductions"`). |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | `algorithm` is not one of the valid choices. |

### Example

```python
from chszlablib import HyperGraph, IndependenceProblems

hg = HyperGraph(num_nodes=6, num_edges=3)
hg.set_edge(0, [0, 1, 2])
hg.set_edge(1, [1, 3, 4])
hg.set_edge(2, [2, 4, 5])
hg.set_edge_weight(0, 10)
hg.set_edge_weight(1, 20)
hg.set_edge_weight(2, 15)
hg.finalize()

result = IndependenceProblems.bmatching(hg, algorithm="greedy_weight_desc")
print(f"Matched {result.num_matched} edges, total weight: {result.total_weight}")

# With custom capacities
hg2 = HyperGraph(num_nodes=6, num_edges=3)
# ... add edges ...
hg2.set_capacity(0, 2)  # vertex 0 can be in up to 2 matched edges
hg2.finalize()
```

---

## `StreamingBMatcher`

True streaming hypergraph matching that processes edges one at a time in a single pass. Suitable for large-scale hypergraphs that do not fit in memory.

### Constructor

```python
StreamingBMatcher(
    num_nodes: int,
    algorithm: str = "greedy",
    capacities: np.ndarray | list[int] | None = None,
    seed: int = 0,
    epsilon: float = 0.0,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | `int` | *required* | Number of vertices in the hypergraph. |
| `algorithm` | `str` | `"greedy"` | Streaming algorithm (see table below). |
| `capacities` | `array-like` or `None` | `None` | Per-vertex capacities. `None` = all capacities are 1. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `epsilon` | `float` | `0.0` | Approximation parameter for greedy. |

#### Available Streaming Algorithms

| Algorithm | Description |
|-----------|-------------|
| `"naive"` | Accept every feasible edge |
| `"greedy_set"` | Greedy with set-based tracking |
| `"best_evict"` | Evict lowest-weight matched edge if beneficial |
| `"greedy"` | Greedy with weight threshold (default, best quality/speed) |
| `"lenient"` | Lenient acceptance threshold |

### Methods

| Method | Description |
|--------|-------------|
| `add_edge(nodes, weight=1.0)` | Feed one hyperedge (list of vertex IDs + weight). |
| `finish()` | Finalize and return a `BMatchingResult`. |
| `reset()` | Reset internal state for re-streaming. |
| `num_edges_streamed` | Property: number of edges fed so far. |

### Example

```python
from chszlablib import StreamingBMatcher

matcher = StreamingBMatcher(num_nodes=100, algorithm="greedy")

# Stream edges one at a time
matcher.add_edge([0, 1, 2], weight=5.0)
matcher.add_edge([3, 4, 5], weight=8.0)
matcher.add_edge([1, 3, 6], weight=3.0)

result = matcher.finish()
print(f"Matched {result.num_matched} edges, total weight: {result.total_weight}")
```

---

## Performance Disclaimer

> This Python interface wraps the HeiHGM C++ libraries via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementations directly from the [HeiHGM/Bmatching](https://github.com/HeiHGM/Bmatching) and [HeiHGM/Streaming](https://github.com/HeiHGM/Streaming) repositories.**

---

## References

- Ernestine Gro{\ss}mann, Henrik M\"uhe, Christian Schulz, and Manuel Penschuck. "Hypergraph B-Matching." *arXiv preprint*, 2024.
