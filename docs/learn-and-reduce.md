# GNN-Guided MWIS Kernelization (LearnAndReduce)

**Original Repository:** [https://github.com/KarlsruheMIS/LearnAndReduce](https://github.com/KarlsruheMIS/LearnAndReduce)

---

## Overview

LearnAndReduce solves the **maximum weight independent set (MWIS)** problem using GNN-guided kernelization. Trained graph neural networks predict which expensive reduction rules will succeed, dramatically speeding up the preprocessing phase. The GNN inference is pure C++ (no PyTorch runtime required), with 7 pre-trained model files shipped inside the package.

The workflow has two stages:

1. **Kernelization**: Apply reduction rules (guided by GNN predictions) to shrink the graph into a smaller kernel. Vertices provably in/out of the optimal solution are fixed, accumulating an offset weight.
2. **Kernel solve + lift**: Solve the (much smaller) kernel with any MWIS solver, then lift the kernel solution back to the original graph.

CHSZLabLib exposes two levels of API:

| API | Description |
|-----|-------------|
| `IndependenceProblems.learn_and_reduce()` | Full pipeline: kernelize + solve + lift (one call) |
| `LearnAndReduceKernel` | Two-step class: separate kernelization and lifting |

---

## `IndependenceProblems.learn_and_reduce()`

Full pipeline: kernelizes the graph, solves the kernel with the chosen solver, and lifts the solution back.

### Signature

```python
IndependenceProblems.learn_and_reduce(
    g: Graph,
    solver: str = "chils",
    config: str = "cyclic_fast",
    gnn_filter: str = "initial_tight",
    time_limit: float = 1000.0,
    solver_time_limit: float = 10.0,
    seed: int = 0,
    num_concurrent: int = 4,
) -> MWISResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `g` | `Graph` | *required* | Input graph. Node weights are **required** and define the objective. |
| `solver` | `str` | `"chils"` | Kernel solver: `"chils"`, `"branch_reduce"`, or `"mmwis"`. |
| `config` | `str` | `"cyclic_fast"` | Reduction preset: `"cyclic_fast"` (degree limit 64) or `"cyclic_strong"` (degree limit 512, more thorough). |
| `gnn_filter` | `str` | `"initial_tight"` | GNN filtering mode: `"initial_tight"`, `"initial"`, `"always"`, or `"never"`. |
| `time_limit` | `float` | `1000.0` | Time limit for the kernelization phase in seconds. |
| `solver_time_limit` | `float` | `10.0` | Time limit for the kernel solver in seconds. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `num_concurrent` | `int` | `4` | Number of concurrent threads (only used when `solver="chils"`). |

### Returns

**`MWISResult`**

| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of vertices in the independent set. |
| `weight` | `int` | Total weight of selected vertices. |
| `vertices` | `np.ndarray` (int32) | Vertex IDs in the independent set. |

### Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Invalid `solver`, `config`, or `gnn_filter`. |

### Example

```python
from chszlablib import Graph, IndependenceProblems

g = Graph.from_metis("weighted_graph.graph")

# Full pipeline with CHILS as kernel solver
result = IndependenceProblems.learn_and_reduce(
    g, solver="chils", time_limit=60.0, solver_time_limit=10.0,
)
print(f"MWIS weight: {result.weight}, size: {result.size}")
print(f"Selected vertices: {result.vertices}")
```

---

## `LearnAndReduceKernel`

Two-step class for separate kernelization and lifting. Useful when you want to inspect the kernel, use a custom solver, or combine with other algorithms.

### Constructor

```python
LearnAndReduceKernel(
    g: Graph,
    config: str = "cyclic_fast",
    gnn_filter: str = "initial_tight",
    time_limit: float = 1000.0,
    seed: int = 0,
)
```

### Methods

#### `kernelize() -> Graph`

Run the GNN-guided reduction rules. Returns the reduced kernel as a `Graph` object (may have 0 nodes if the instance is fully reduced).

#### `lift_solution(kernel_vertices: np.ndarray) -> MWISResult`

Map a kernel independent set back to the original graph. `kernel_vertices` is a 1-D int array of vertex IDs in the kernel graph.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `offset_weight` | `int` | Weight determined by reductions alone (before solving kernel). |
| `kernel_nodes` | `int` | Number of nodes in the reduced kernel (-1 if not yet kernelized). |

### Example

```python
from chszlablib import Graph, IndependenceProblems, LearnAndReduceKernel
import numpy as np

g = Graph.from_metis("weighted_graph.graph")

# Step 1: Kernelize
lr = LearnAndReduceKernel(g, config="cyclic_fast")
kernel = lr.kernelize()
print(f"Kernel: {lr.kernel_nodes} nodes (from {g.num_nodes}), offset: {lr.offset_weight}")

# Step 2: Solve kernel (use any solver)
if lr.kernel_nodes > 0:
    sol = IndependenceProblems.chils(kernel, time_limit=10.0)
    result = lr.lift_solution(sol.vertices)
else:
    result = lr.lift_solution(np.array([], dtype=np.int32))

print(f"MWIS weight: {result.weight}, size: {result.size}")
```

---

## Configuration Options

### Reduction Presets (`config`)

| Config | Degree Limit | Set Limit | Description |
|--------|-------------|-----------|-------------|
| `cyclic_fast` | 64 | 512 | Fast reductions, good for large graphs |
| `cyclic_strong` | 512 | 2048 | More thorough, smaller kernels |

### GNN Filter Modes (`gnn_filter`)

| Mode | Description |
|------|-------------|
| `initial_tight` | GNN filters only on first round, tight threshold (default) |
| `initial` | GNN filters only on first round |
| `always` | GNN filters on every round |
| `never` | Disable GNN filtering (classical reductions only) |

### Kernel Solvers (`solver`)

| Solver | Type | Description |
|--------|------|-------------|
| `chils` | Heuristic | Concurrent local search (default, best for large kernels) |
| `branch_reduce` | Exact | Branch-and-reduce (optimal, exponential worst-case) |
| `mmwis` | Heuristic | Memetic evolutionary algorithm |

---

## Performance Disclaimer

> This Python interface wraps the LearnAndReduce C++ library via pybind11. While convenient for prototyping and integration into Python workflows, there is inherent overhead from the Python/C++ boundary and data conversion. **For maximum performance on large-scale instances, use the original C++ implementation directly from the [LearnAndReduce repository](https://github.com/KarlsruheMIS/LearnAndReduce).**

---

## References

```bibtex
@inproceedings{grossmann2025reductions,
  author    = {Ernestine Gro{\ss}mann and Kenneth Langedal and Christian Schulz},
  title     = {Accelerating Reductions Using Graph Neural Networks for the Maximum
               Weight Independent Set Problem},
  booktitle = {Conference on Applied and Computational Discrete Algorithms ({ACDA})},
  pages     = {155--168},
  publisher = {SIAM},
  year      = {2025},
  doi       = {10.1137/1.9781611978759.12}
}
```
