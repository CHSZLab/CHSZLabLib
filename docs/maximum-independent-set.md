# Maximum Independent Set (KaMIS)

**Original Repository:** [https://github.com/KarlsruheMIS/KaMIS](https://github.com/KarlsruheMIS/KaMIS)

---

## Overview

KaMIS (Karlsruhe Maximum Independent Sets) provides a suite of algorithms for the **maximum independent set (MIS)** and **maximum weight independent set (MWIS)** problems. Given an undirected graph *G = (V, E)*, find an independent set *I* (a set of pairwise non-adjacent vertices) that maximizes either:

- **Cardinality** (MIS): maximize |I|
- **Total weight** (MWIS): maximize the sum of node weights in I

Both problems are NP-hard and hard to approximate. KaMIS combines powerful **graph reduction rules** (crown, LP, domination, twin) with evolutionary and local search algorithms.

CHSZLabLib exposes four KaMIS solvers:

| Method | Problem | Type | Description |
|--------|---------|------|-------------|
| `redumis()` | MIS | Heuristic | Evolutionary algorithm on reduced kernel |
| `online_mis()` | MIS | Heuristic | Iterated local search (faster, smaller sets) |
| `branch_reduce()` | MWIS | Exact | Branch-and-reduce (optimal, exponential worst-case) |
| `mmwis()` | MWIS | Heuristic | Memetic evolutionary algorithm |

---

## `IndependenceProblems.redumis()`

Maximum cardinality independent set via evolutionary algorithm with graph reductions.

### Signature

```python
IndependenceProblems.redumis(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
) -> MISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph (unweighted; node weights are forwarded but not used for the objective). |
| `time_limit` | `float` | `10.0` | Wall-clock time budget in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`MISResult`**

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of vertices in the independent set. |
| `weight` | `int` | Total weight of selected vertices. |
| `vertices` | `np.ndarray` (int32) | Vertex IDs in the independent set. |

---

## `IndependenceProblems.online_mis()`

Maximum cardinality independent set via iterated local search. Faster than ReduMIS but generally produces smaller sets.

### Signature

```python
IndependenceProblems.online_mis(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
    ils_iterations: int = 15000,
) -> MISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph. |
| `time_limit` | `float` | `10.0` | Wall-clock time budget in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `ils_iterations` | `int` | `15000` | Number of iterated local search iterations. |

### Returns

**`MISResult`** (same fields as above).

---

## `IndependenceProblems.branch_reduce()`

**Exact** maximum weight independent set via branch-and-reduce. Guaranteed to find an optimal solution but may require exponential time.

### Signature

```python
IndependenceProblems.branch_reduce(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
) -> MISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph. Node weights define the objective; if unset, all weights default to 1 (equivalent to unweighted MIS). |
| `time_limit` | `float` | `10.0` | Wall-clock time budget in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`MISResult`** (same fields as above).

---

## `IndependenceProblems.mmwis()`

Approximate maximum weight independent set via memetic evolutionary algorithm. Trades exactness for scalability on larger instances.

### Signature

```python
IndependenceProblems.mmwis(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
) -> MISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph. Node weights define the objective; if unset, all weights default to 1. |
| `time_limit` | `float` | `10.0` | Wall-clock time budget in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |

### Returns

**`MISResult`** (same fields as above).

---

## Common Exceptions (All Methods)

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `time_limit < 0`. |

---

## Example

```python
from chszlablib import Graph, IndependenceProblems

g = Graph.from_metis("graph.graph")

# Unweighted MIS: evolutionary approach
mis = IndependenceProblems.redumis(g, time_limit=30.0)
print(f"MIS size: {mis.size}, vertices: {mis.vertices}")

# Unweighted MIS: fast local search
mis_fast = IndependenceProblems.online_mis(g, time_limit=5.0)
print(f"Fast MIS size: {mis_fast.size}")

# Weighted MIS: exact solver (small graphs)
g_weighted = Graph.from_metis("weighted_graph.graph")
exact = IndependenceProblems.branch_reduce(g_weighted, time_limit=60.0)
print(f"Exact MWIS weight: {exact.weight}")

# Weighted MIS: evolutionary approach (large graphs)
approx = IndependenceProblems.mmwis(g_weighted, time_limit=30.0)
print(f"Approx MWIS weight: {approx.weight}")
```

---

## Performance Disclaimer

> This Python interface wraps the KaMIS C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances or in production HPC pipelines, use the original C++ implementation directly from the [KaMIS repository](https://github.com/KarlsruheMIS/KaMIS).**

---

## References

- Sebastian Lamm, Christian Schulz, Darren Strash, Robert Williger, and Huashuo Zhang. "Exactly Solving the Maximum Weight Independent Set Problem on Large Real-World Graphs." *Proceedings of the 21st Workshop on Algorithm Engineering and Experiments (ALENEX)*, 2019.
- Demian Hespe, Sebastian Lamm, Christian Schulz, and Darren Strash. "WeGotYouCovered: The Winning Solver from the PACE 2019 Challenge, Vertex Cover Track." *CSC*, 2020.
