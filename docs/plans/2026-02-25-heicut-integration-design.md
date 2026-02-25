# HeiCut Integration Design

## Goal

Integrate HeiCut (exact minimum cut solver for hypergraphs) into CHSZLabLib as `Decomposition.hypermincut()`, exposing all 4 solver methods: kernelizer, submodular, ILP, and trimmer.

## Problem

Given a hypergraph H = (V, E) with edge weights w: E -> N, find a partition of V into two non-empty sets S and V\S minimizing the total weight of hyperedges that intersect both S and V\S.

## API

```python
@dataclass
class HyperMincutResult:
    cut_value: int   # Exact minimum cut value
    time: float      # Computation time (seconds)
    method: str      # Solver used

Decomposition.hypermincut(
    hg: HyperGraph,
    method: Literal["kernelizer", "submodular", "ilp", "trimmer"] = "kernelizer",
    time_limit: float = 300.0,
    seed: int = 0,
    num_threads: int = 1,
) -> HyperMincutResult
```

### Methods

| Method | Description | Gurobi? | Weighted? |
|--------|-------------|---------|-----------|
| `kernelizer` | Full HeiCut: reductions + submodular base solver | No | Yes |
| `submodular` | Vertex-ordering solver (Tight ordering) | No | Yes |
| `ilp` | Relaxed-BIP ILP solver | Yes (gurobipy fallback) | Yes |
| `trimmer` | Chekuri-Xu trimmer certificates | No | No (unweighted) |

## Architecture

### Build chain

1. **Mt-KaHyPar** — git submodule at `external_repositories/mt-kahypar` (commit 0ef674a). Built from source via CMake. Produces `libmtkahypar.so`. System deps: TBB, Boost, hwloc.
2. **HeiCut** — git submodule at `external_repositories/HeiCut`. Compiled as `heicut_static` from `lib/` sources. Links Mt-KaHyPar headers.
3. **`_heicut` pybind11 module** — `bindings/heicut_binding.cpp`. Links `heicut_static` + `libmtkahypar.so`.

### Gurobi handling

- CMake checks for Gurobi at build time
- If found: `ilp.cpp` compiled into `heicut_static`, native ILP available
- If not found: `ilp.cpp` skipped
- Python fallback: `method="ilp"` uses gurobipy if available (same pattern as HyperMIS)
- If neither C++ Gurobi nor gurobipy available: `method="ilp"` raises `ImportError`

### Data flow

```
HyperGraph (Python CSR)
  → eptr/everts/edge_weights numpy arrays
  → pybind11 binding
  → mt_kahypar_create_hypergraph() builds StaticHypergraph
  → Solver runs on StaticHypergraph
  → Returns (cut_value, time)
  → HyperMincutResult
```

### C++ binding functions

```
py_heicut_kernelizer(eptr, everts, edge_weights, num_nodes, ordering_type, lp_iterations, seed, num_threads, verbose) → (cut_value, time)
py_heicut_submodular(eptr, everts, edge_weights, num_nodes, ordering_type, seed, num_threads) → (cut_value, time)
py_heicut_ilp(eptr, everts, edge_weights, num_nodes, ilp_timeout, seed, num_threads) → (cut_value, time, is_optimal)
py_heicut_trimmer(eptr, everts, num_nodes, ordering_type, seed) → (cut_value, time)
```

## Files

| File | Action |
|------|--------|
| `external_repositories/HeiCut/` | Add git submodule |
| `external_repositories/mt-kahypar/` | Add git submodule (commit 0ef674a) |
| `bindings/heicut_binding.cpp` | Create |
| `chszlablib/decomposition.py` | Modify: add `hypermincut()`, `HyperMincutResult` |
| `chszlablib/_gurobi_hypermincut_ilp.py` | Create: Python gurobipy ILP fallback |
| `chszlablib/__init__.py` | Modify: export `HyperMincutResult` |
| `CMakeLists.txt` | Modify: Mt-KaHyPar, HeiCut, `_heicut` module |
| `tests/test_hypermincut.py` | Create |
| `README.md` | Modify |

## Testing

- Each method on small known hypergraphs (disjoint edges, single edge, triangle)
- ILP method raises ImportError without Gurobi
- Trimmer on unweighted hypergraphs only
- Invalid method raises InvalidModeError
- Edge cases: single vertex, disconnected components
