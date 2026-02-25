# HyperGraph Data Structure + HyperMIS Integration

**Date:** 2026-02-25
**Status:** Approved

## Overview

Add a `HyperGraph` data structure to CHSZLabLib and integrate the HyperMIS library (Maximum Independent Set on Hypergraphs) into the `IndependenceProblems` namespace. The HyperGraph class is designed to be reusable for future hypergraph algorithms.

## HyperGraph Class (`chszlablib/hypergraph.py`)

### Internal Representation: Dual CSR

Two CSR-style array pairs encoding the bipartite relationship between vertices and hyperedges:

- `vptr` (int64, length n+1): vertex-to-edge index pointers
- `vedges` (int32): concatenated hyperedge IDs per vertex
- `eptr` (int64, length m+1): edge-to-vertex index pointers
- `everts` (int32): concatenated vertex IDs per hyperedge
- `node_weights` (int64, length n): default all-ones
- `edge_weights` (int64, length m): default all-ones

### Builder API

```python
# Option A: vertex-by-vertex
hg = HyperGraph(num_nodes=10, num_edges=5)
hg.add_to_edge(edge_id=0, vertex=3)
hg.add_to_edge(edge_id=0, vertex=7)
hg.finalize()

# Option B: whole-edge at once
hg = HyperGraph(num_nodes=10, num_edges=3)
hg.set_edge(0, [3, 7, 9])
hg.set_edge(1, [1, 2])
hg.finalize()
```

Follows the same pattern as `Graph`: builder state pre-finalize, immutable CSR post-finalize, idempotent `finalize()`.

### Batch Constructors

- `HyperGraph.from_edge_list(edges: list[list[int]], num_nodes=None)` — each inner list is a hyperedge
- `HyperGraph.from_hmetis(path)` — reads hMETIS format file
- `HyperGraph.from_dual_csr(vptr, vedges, eptr, everts, node_weights=None, edge_weights=None)` — direct CSR construction

### Export Methods

- `to_hmetis(path)` — writes hMETIS format
- `to_graph()` — clique expansion (each hyperedge becomes a clique), returns `Graph`

### Properties (auto-finalize)

`num_nodes`, `num_edges`, `vptr`, `vedges`, `eptr`, `everts`, `node_weights`, `edge_weights`

### Validation

- Vertex IDs in `[0, num_nodes)`, edge IDs in `[0, num_edges)`
- No duplicate vertex in same hyperedge
- Hyperedges must have >= 1 vertex
- Raises `InvalidHyperGraphError`

## HyperMIS Integration

### Result Type

```python
@dataclass
class HyperMISResult:
    size: int                  # Cardinality of independent set
    weight: int                # Total node weight of selected vertices
    vertices: np.ndarray       # 1-D int array of vertex IDs
    offset: int                # Vertices determined during reduction
    reduction_time: float      # Time spent on reductions (seconds)
```

### Method Signature

```python
IndependenceProblems.hypermis(
    hg: HyperGraph,
    time_limit: float = 60.0,
    reduction_time_limit: float = 50.0,
    seed: int = 0,
    strong_reductions: bool = False,
) -> HyperMISResult
```

### Gurobi Handling

- CMake: `find_package(Gurobi QUIET)` — conditional compilation
- Two pybind11 modules:
  - `_hypermis` (always built): reduction-only
  - `_hypermis_ilp` (Gurobi required): reductions + ILP solver
- Python: `IndependenceProblems.HYPERMIS_ILP_AVAILABLE: bool` for introspection
- Without Gurobi: `hypermis()` uses reduction-only mode
- With Gurobi: full reductions + ILP

## pybind11 Bindings

### `_hypermis` (always built)

Input: eptr, everts, node_weights, edge_weights, config params
Output: (offset, is_vertices, reduction_time)

### `_hypermis_ilp` (Gurobi only)

Input: eptr, everts, node_weights, edge_weights, config params, time limits
Output: (is_size, is_vertices, offset, reduction_time)

## hMETIS I/O

Add to `chszlablib/io.py`:

- `read_hmetis(path) -> HyperGraph`
- `write_hmetis(hg: HyperGraph, path) -> None`

Format: header `M N [W]`, M edge lines (1-indexed vertices), optional N vertex weight lines.

## New Exception

`InvalidHyperGraphError(CHSZLabLibError, ValueError)` in `exceptions.py`.

## Build System

- Git submodule: `external_repositories/HyperMIS`
- Static library `hypermis_static` from HyperMIS sources (hypergraph.cpp, reductions.cpp, config.cpp)
- Conditional Gurobi linking for ILP module

## Public API Exports

- `HyperGraph`, `HyperMISResult`, `InvalidHyperGraphError`
- `read_hmetis`, `write_hmetis`
- Updated `describe()` function

## Testing

- `test_hypergraph.py`: construction, dual CSR, validation, hMETIS I/O, clique expansion
- `test_hypermis.py`: small instances, independent set validity, reduction-only mode
