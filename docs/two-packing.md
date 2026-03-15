# Maximum 2-Packing Set (red2pack)

**Original Repository:** [https://github.com/KarlsruheMIS/red2pack](https://github.com/KarlsruheMIS/red2pack)

---

## Overview

red2pack solves the **maximum (weighted) 2-packing set problem**: given a graph $G = (V, E)$, find a largest-weight subset $S \subseteq V$ such that no two vertices in $S$ share a common neighbor, i.e.,

$$\max_{S \subseteq V} \sum_{v \in S} c(v) \quad \text{subject to} \quad \text{dist}(u, v) \geq 3 \quad \text{for all } u, v \in S, u \neq v.$$

A 2-packing set is also known as a **distance-3 independent set**. Applications include facility placement (no two facilities share a customer), wireless channel assignment, and domination problems in networks.

The solver uses a **reduce-and-transform** strategy:

1. **Reduce**: Apply problem-specific reduction rules that shrink the graph while preserving the optimal solution.
2. **Transform**: Convert the reduced 2-packing instance into an equivalent **maximum weight independent set (MWIS)** problem on a (typically much smaller) graph.
3. **Solve**: Solve the MWIS kernel with one of several backend solvers (exact or heuristic).
4. **Lift**: Map the MWIS solution back to a 2-packing set in the original graph.

CHSZLabLib exposes two levels of API:

| API | Description |
|-----|-------------|
| `IndependenceProblems.two_packing()` | Full pipeline: reduce + transform + solve + lift (one call) |
| `TwoPackingKernel` | Two-step class: separate reduction/transformation and lifting |

---

## `IndependenceProblems.two_packing()`

Full pipeline: reduces the graph, transforms to MIS, solves with the chosen algorithm, and lifts the solution back.

### Signature

```python
IndependenceProblems.two_packing(
    g: Graph,
    algorithm: str = "chils",
    time_limit: float = 100.0,
    seed: int = 0,
    reduction_style: str = "",
) -> TwoPackingResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph. Node weights define the objective; if unset, all weights default to 1. |
| `algorithm` | `str` | `"chils"` | Algorithm for solving the transformed MIS kernel (see table below). |
| `time_limit` | `float` | `100.0` | Wall-clock time budget in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `reduction_style` | `str` | `""` | Reduction preset: `""` (default), `"fast"`, `"strong"`, `"full"`, `"heuristic"`. |

### Returns

**`TwoPackingResult`**

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of vertices in the 2-packing set. |
| `weight` | `int` | Total weight of selected vertices. |
| `vertices` | `np.ndarray` (int32) | Vertex IDs in the 2-packing set. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `InvalidModeError` | Invalid `algorithm` string. |
| `ValueError` | Negative `time_limit`. |

### Example

```python
from chszlablib import Graph, IndependenceProblems

# Path graph: 0-1-2-3-4
g = Graph.from_edge_list([(0,1),(1,2),(2,3),(3,4)])

# Default algorithm (CHILS)
result = IndependenceProblems.two_packing(g, algorithm="chils", time_limit=10.0)
print(f"2-packing size: {result.size}, vertices: {result.vertices}")

# Exact solver
result = IndependenceProblems.two_packing(g, algorithm="exact")
print(f"Exact 2-packing size: {result.size}")

# Weighted graph
g = Graph(num_nodes=5)
for u, v in [(0,1),(1,2),(2,3),(3,4)]:
    g.add_edge(u, v)
for i, w in enumerate([10, 1, 1, 1, 10]):
    g.set_node_weight(i, w)
result = IndependenceProblems.two_packing(g, algorithm="chils")
print(f"Weight: {result.weight}, vertices: {result.vertices}")
```

---

## `TwoPackingKernel`

Two-step class for separate reduction/transformation and lifting. Useful when you want to inspect the kernel, use a custom solver, or combine with other algorithms (e.g., solve the MIS kernel with an ILP solver).

### Constructor

```python
TwoPackingKernel(
    g: Graph,
    reduction_style: str = "",
    time_limit: float = 1000.0,
    seed: int = 0,
    weighted: bool | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph with optional node weights. |
| `reduction_style` | `str` | `""` | Reduction preset: `""`, `"fast"`, `"strong"`, `"full"`, `"heuristic"`. |
| `time_limit` | `float` | `1000.0` | Time limit for the reduction phase in seconds. |
| `seed` | `int` | `0` | Random seed. |
| `weighted` | `bool \| None` | `None` | Use weighted reductions. `None` auto-detects from `g.node_weights`. |

### Methods

#### `reduce_and_transform() -> Graph`

Run the 2-packing reduction rules and transform the remaining instance into an equivalent MIS problem. Returns the kernel as a `Graph` object (may have 0 nodes if the instance is fully reduced by reductions alone).

#### `lift_solution(mis_vertices: np.ndarray) -> TwoPackingResult`

Map a kernel MIS solution back to the original 2-packing set. `mis_vertices` is a 1-D int array of vertex IDs in the kernel graph (0-indexed).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `offset_weight` | `int` | Weight determined by reductions alone (before solving kernel). |
| `kernel_nodes` | `int` | Number of nodes in the reduced kernel (-1 if not yet reduced). |

### Example

```python
from chszlablib import Graph, IndependenceProblems, TwoPackingKernel
import numpy as np

g = Graph.from_metis("large_graph.graph")

# Step 1: Reduce and transform to MIS kernel
tpk = TwoPackingKernel(g)
kernel = tpk.reduce_and_transform()
print(f"Kernel: {tpk.kernel_nodes} nodes (from {g.num_nodes}), offset: {tpk.offset_weight}")

# Step 2: Solve kernel with any MIS/MWIS solver
if tpk.kernel_nodes > 0:
    sol = IndependenceProblems.branch_reduce(kernel, time_limit=60.0)
    result = tpk.lift_solution(sol.vertices)
else:
    result = tpk.lift_solution(np.array([], dtype=np.int32))

print(f"2-packing weight: {result.weight}, size: {result.size}")
print(f"Vertices: {result.vertices}")
```

---

## Algorithms

All algorithms first apply the reduce-and-transform step, then solve the resulting MIS kernel with different backend solvers.

| Algorithm | Type | Description |
|-----------|------|-------------|
| `"exact"` | Exact (unweighted) | Branch-and-reduce for maximum cardinality (KaMIS) |
| `"exact_weighted"` | Exact (weighted) | Branch-and-reduce for maximum weight (KaMIS) |
| `"chils"` | Heuristic (weighted) | Concurrent local search (CHILS) — default, best general-purpose |
| `"drp"` | Heuristic (weighted) | Dynamic reduce-and-peel |
| `"htwis"` | Heuristic (weighted) | Heavy-tailed weighted independent set |
| `"hils"` | Heuristic (weighted) | Heavy independent local search |
| `"mmwis"` | Heuristic (weighted) | Memetic evolutionary algorithm (KaMIS) |
| `"online"` | Heuristic (unweighted) | Iterated local search (KaMIS OnlineMIS) |
| `"ilp"` | Exact (weighted) | Kernelize + ILP on kernel (requires `gurobipy`) |

> **Note:** The `"ilp"` algorithm uses `TwoPackingKernel` for reduction, then formulates a maximum weight independent set ILP on the kernel graph. This requires `pip install gurobipy` and a valid [Gurobi license](https://www.gurobi.com/downloads/).

### Available Constants

```python
IndependenceProblems.TWO_PACKING_ALGORITHMS
# ("exact", "exact_weighted", "chils", "drp", "htwis", "hils", "mmwis", "online", "ilp")
```

---

## Reduction Styles

| Style | Description |
|-------|-------------|
| `""` | Default reductions (from red2pack configurator) |
| `"fast"` | Fast reductions only |
| `"strong"` | Stronger reductions, smaller kernels |
| `"full"` | All available reductions |
| `"heuristic"` | Heuristic reductions |

---

## Performance Disclaimer

> This Python interface wraps the red2pack C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [red2pack repository](https://github.com/KarlsruheMIS/red2pack).**

---

## References

```bibtex
@article{borowitz2025scalable,
  author    = {Jannick Borowitz and Ernestine Gro{\ss}mann and Christian Schulz and Dominik Schweisgut},
  title     = {Scalable Algorithms for 2-Packing Sets on Arbitrary Graphs},
  journal   = {Journal of Graph Algorithms and Applications},
  volume    = {29},
  number    = {1},
  pages     = {159--186},
  year      = {2025},
  doi       = {10.7155/jgaa.v29i1.3064}
}

@article{borowitz2025weighted2packing,
  author    = {Jannick Borowitz and Ernestine Gro{\ss}mann and Christian Schulz},
  title     = {Finding Maximum Weight 2-Packing Sets on Arbitrary Graphs},
  journal   = {Networks},
  year      = {2025},
  doi       = {10.1002/net.70028}
}
```
