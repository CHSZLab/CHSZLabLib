# HyperGraph + HyperMIS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable `HyperGraph` data structure and integrate the HyperMIS library for Maximum Independent Set on hypergraphs.

**Architecture:** New `HyperGraph` class in `chszlablib/hypergraph.py` with dual-CSR representation (vertex-to-edge + edge-to-vertex). HyperMIS C++ code compiled as static library, wrapped via pybind11 into `_hypermis` (reductions, always built) and `_hypermis_ilp` (ILP solver, Gurobi optional). Python method `IndependenceProblems.hypermis()` dispatches to the appropriate binding.

**Tech Stack:** Python 3, NumPy, pybind11, CMake, C++17, Gurobi (optional)

**Design doc:** `docs/plans/2026-02-25-hypergraph-hypermis-design.md`

---

### Task 1: Add InvalidHyperGraphError exception

**Files:**
- Modify: `chszlablib/exceptions.py`
- Test: `tests/test_hypergraph.py` (new file, started here)

**Step 1: Write the failing test**

Create `tests/test_hypergraph.py`:

```python
"""Tests for the HyperGraph class."""

import pytest
from chszlablib.exceptions import InvalidHyperGraphError, CHSZLabLibError


class TestInvalidHyperGraphError:
    def test_is_chszlablib_error(self):
        with pytest.raises(CHSZLabLibError):
            raise InvalidHyperGraphError("test")

    def test_is_value_error(self):
        with pytest.raises(ValueError):
            raise InvalidHyperGraphError("test")

    def test_message(self):
        with pytest.raises(InvalidHyperGraphError, match="bad hypergraph"):
            raise InvalidHyperGraphError("bad hypergraph")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypergraph.py::TestInvalidHyperGraphError -v`
Expected: FAIL with `ImportError: cannot import name 'InvalidHyperGraphError'`

**Step 3: Write minimal implementation**

Add to `chszlablib/exceptions.py` after `InvalidGraphError`:

```python
class InvalidHyperGraphError(CHSZLabLibError, ValueError):
    """Raised when a hypergraph has invalid structure."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hypergraph.py::TestInvalidHyperGraphError -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add chszlablib/exceptions.py tests/test_hypergraph.py
git commit -m "feat: add InvalidHyperGraphError exception"
```

---

### Task 2: HyperGraph core — constructor and finalize with dual CSR

**Files:**
- Create: `chszlablib/hypergraph.py`
- Test: `tests/test_hypergraph.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_hypergraph.py`:

```python
import numpy as np
from chszlablib.hypergraph import HyperGraph


class TestHyperGraphConstruction:
    def test_basic_construction(self):
        """Triangle hypergraph: edge 0={0,1,2}, edge 1={1,2,3}."""
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [1, 2, 3])
        hg.finalize()
        assert hg.num_nodes == 4
        assert hg.num_edges == 2

    def test_eptr_everts(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        hg.set_edge(1, [1, 2])
        hg.finalize()
        # eptr: edge 0 has 2 verts, edge 1 has 2 verts
        np.testing.assert_array_equal(hg.eptr, [0, 2, 4])
        # everts sorted within each edge
        np.testing.assert_array_equal(hg.everts, [0, 1, 1, 2])

    def test_vptr_vedges(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        hg.set_edge(1, [1, 2])
        hg.finalize()
        # vertex 0 in edge 0; vertex 1 in edges 0,1; vertex 2 in edge 1
        np.testing.assert_array_equal(hg.vptr, [0, 1, 3, 4])
        np.testing.assert_array_equal(hg.vedges, [0, 0, 1, 1])

    def test_add_to_edge(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.add_to_edge(0, 0)
        hg.add_to_edge(0, 2)
        hg.finalize()
        np.testing.assert_array_equal(hg.eptr, [0, 2])
        np.testing.assert_array_equal(hg.everts, [0, 2])

    def test_auto_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        # Accessing property should auto-finalize
        assert hg.num_nodes == 3
        assert hg.num_edges == 1
        assert len(hg.everts) == 3

    def test_finalize_idempotent(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        hg.finalize()  # Should not raise
        assert hg.num_edges == 1

    def test_default_weights(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        hg.finalize()
        np.testing.assert_array_equal(hg.node_weights, [1, 1, 1])
        np.testing.assert_array_equal(hg.edge_weights, [1])


class TestHyperGraphWeights:
    def test_node_weights(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        hg.set_node_weight(0, 5)
        hg.set_node_weight(2, 10)
        hg.finalize()
        np.testing.assert_array_equal(hg.node_weights, [5, 1, 10])

    def test_edge_weights(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        hg.set_edge(1, [1, 2])
        hg.set_edge_weight(0, 3)
        hg.set_edge_weight(1, 7)
        hg.finalize()
        np.testing.assert_array_equal(hg.edge_weights, [3, 7])


class TestHyperGraphValidation:
    def test_negative_num_nodes(self):
        with pytest.raises(InvalidHyperGraphError):
            HyperGraph(num_nodes=-1, num_edges=1)

    def test_negative_num_edges(self):
        with pytest.raises(InvalidHyperGraphError):
            HyperGraph(num_nodes=1, num_edges=-1)

    def test_vertex_out_of_range(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError):
            hg.add_to_edge(0, 5)

    def test_edge_out_of_range(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError):
            hg.add_to_edge(2, 0)

    def test_duplicate_vertex_in_edge(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.add_to_edge(0, 1)
        with pytest.raises(InvalidHyperGraphError):
            hg.add_to_edge(0, 1)

    def test_modify_after_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        with pytest.raises(GraphNotFinalizedError):
            hg.add_to_edge(0, 2)

    def test_empty_edge_raises(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        # Don't add any vertices to edge 0
        with pytest.raises(InvalidHyperGraphError):
            hg.finalize()


class TestHyperGraphRepr:
    def test_finalized(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.finalize()
        r = repr(hg)
        assert "n=4" in r
        assert "m=2" in r

    def test_not_finalized(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        r = repr(hg)
        assert "finalized=False" in r
```

Add import of `GraphNotFinalizedError` at top:
```python
from chszlablib.exceptions import InvalidHyperGraphError, CHSZLabLibError, GraphNotFinalizedError
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hypergraph.py -v -k "not TestInvalidHyperGraphError"`
Expected: FAIL with `ImportError: cannot import name 'HyperGraph'`

**Step 3: Write implementation**

Create `chszlablib/hypergraph.py` with:
- `HyperGraph.__init__(num_nodes, num_edges)` — validates, initializes builder state
- `add_to_edge(edge_id, vertex)` — adds one vertex to one edge
- `set_edge(edge_id, vertices)` — sets all vertices of an edge at once (clears previous)
- `set_node_weight(node, weight)` / `set_edge_weight(edge, weight)`
- `finalize()` — builds dual CSR arrays (eptr/everts + vptr/vedges), validates no empty edges
- Properties: `num_nodes`, `num_edges`, `eptr`, `everts`, `vptr`, `vedges`, `node_weights`, `edge_weights`
- `__repr__`

Builder state (pre-finalize):
- `_edge_contents: list[list[int]]` — one list per edge
- `_edge_vertex_sets: list[set[int]]` — for duplicate detection
- `_node_weight_map: dict[int, int]`
- `_edge_weight_map: dict[int, int]`

Finalize algorithm:
1. Validate all edges have >= 1 vertex
2. Sort vertices within each edge
3. Build `eptr`/`everts` from `_edge_contents`
4. Build `vptr`/`vedges` by iterating edges and accumulating per-vertex edge lists
5. Build weight arrays

**Step 4: Run tests**

Run: `pytest tests/test_hypergraph.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add chszlablib/hypergraph.py tests/test_hypergraph.py
git commit -m "feat: add HyperGraph class with dual CSR and builder API"
```

---

### Task 3: HyperGraph batch constructors (from_edge_list, from_dual_csr)

**Files:**
- Modify: `chszlablib/hypergraph.py`
- Test: `tests/test_hypergraph.py` (append)

**Step 1: Write failing tests**

```python
class TestHyperGraphFromEdgeList:
    def test_basic(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3]])
        assert hg.num_nodes == 4
        assert hg.num_edges == 2
        np.testing.assert_array_equal(hg.eptr, [0, 3, 5])

    def test_explicit_num_nodes(self):
        hg = HyperGraph.from_edge_list([[0, 1]], num_nodes=10)
        assert hg.num_nodes == 10

    def test_infer_num_nodes(self):
        hg = HyperGraph.from_edge_list([[0, 5], [3, 7]])
        assert hg.num_nodes == 8  # max vertex ID + 1

    def test_empty_edge_list(self):
        hg = HyperGraph.from_edge_list([], num_nodes=0)
        assert hg.num_nodes == 0
        assert hg.num_edges == 0

    def test_with_node_weights(self):
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2]], node_weights=[5, 3, 7])
        np.testing.assert_array_equal(hg.node_weights, [5, 3, 7])

    def test_with_edge_weights(self):
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2]], edge_weights=[10, 20])
        np.testing.assert_array_equal(hg.edge_weights, [10, 20])


class TestHyperGraphFromDualCSR:
    def test_basic(self):
        # 2 edges: edge 0={0,1}, edge 1={1,2}
        eptr = np.array([0, 2, 4], dtype=np.int64)
        everts = np.array([0, 1, 1, 2], dtype=np.int32)
        vptr = np.array([0, 1, 3, 4], dtype=np.int64)
        vedges = np.array([0, 0, 1, 1], dtype=np.int32)
        hg = HyperGraph.from_dual_csr(vptr, vedges, eptr, everts)
        assert hg.num_nodes == 3
        assert hg.num_edges == 2

    def test_with_weights(self):
        eptr = np.array([0, 2], dtype=np.int64)
        everts = np.array([0, 1], dtype=np.int32)
        vptr = np.array([0, 1, 2], dtype=np.int64)
        vedges = np.array([0, 0], dtype=np.int32)
        nw = np.array([5, 10], dtype=np.int64)
        ew = np.array([3], dtype=np.int64)
        hg = HyperGraph.from_dual_csr(vptr, vedges, eptr, everts,
                                       node_weights=nw, edge_weights=ew)
        np.testing.assert_array_equal(hg.node_weights, [5, 10])
        np.testing.assert_array_equal(hg.edge_weights, [3])

    def test_invalid_eptr(self):
        eptr = np.array([1, 2], dtype=np.int64)  # eptr[0] != 0
        everts = np.array([0], dtype=np.int32)
        vptr = np.array([0, 1], dtype=np.int64)
        vedges = np.array([0], dtype=np.int32)
        with pytest.raises(InvalidHyperGraphError):
            HyperGraph.from_dual_csr(vptr, vedges, eptr, everts)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hypergraph.py -v -k "FromEdgeList or FromDualCSR"`
Expected: FAIL

**Step 3: Implement `from_edge_list` and `from_dual_csr`**

`from_edge_list(edges, num_nodes=None, node_weights=None, edge_weights=None)`:
- Infer `num_nodes` from max vertex ID + 1 if not given
- Create `HyperGraph`, call `set_edge` for each, optionally set weights, finalize

`from_dual_csr(vptr, vedges, eptr, everts, node_weights=None, edge_weights=None)`:
- Validate shapes: `eptr[0]==0`, `eptr[-1]==len(everts)`, monotonic, bounds
- Same for vptr/vedges
- Use `cls.__new__()` pattern (same as `Graph.from_csr`)

**Step 4: Run tests**

Run: `pytest tests/test_hypergraph.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add chszlablib/hypergraph.py tests/test_hypergraph.py
git commit -m "feat: add HyperGraph.from_edge_list and from_dual_csr constructors"
```

---

### Task 4: hMETIS I/O (read_hmetis, write_hmetis)

**Files:**
- Modify: `chszlablib/io.py`
- Modify: `chszlablib/hypergraph.py` (add `from_hmetis`, `to_hmetis`)
- Test: `tests/test_hypergraph.py` (append)

**Step 1: Write failing tests**

```python
import tempfile
import os


class TestHMetisIO:
    def test_write_read_roundtrip_unweighted(self, tmp_path):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3], [0, 3]])
        path = str(tmp_path / "test.hgr")
        hg.to_hmetis(path)
        hg2 = HyperGraph.from_hmetis(path)
        assert hg2.num_nodes == 4
        assert hg2.num_edges == 3
        np.testing.assert_array_equal(hg2.eptr, hg.eptr)
        np.testing.assert_array_equal(hg2.everts, hg.everts)

    def test_write_read_roundtrip_weighted(self, tmp_path):
        hg = HyperGraph.from_edge_list(
            [[0, 1], [1, 2]],
            node_weights=[5, 3, 7],
            edge_weights=[10, 20],
        )
        path = str(tmp_path / "test_w.hgr")
        hg.to_hmetis(path)
        hg2 = HyperGraph.from_hmetis(path)
        np.testing.assert_array_equal(hg2.node_weights, [5, 3, 7])
        np.testing.assert_array_equal(hg2.edge_weights, [10, 20])

    def test_read_hmetis_format(self, tmp_path):
        """Test reading a hand-written hMETIS file."""
        path = str(tmp_path / "manual.hgr")
        with open(path, "w") as f:
            f.write("3 4\n")       # 3 edges, 4 vertices, unweighted
            f.write("1 2 3\n")     # edge 0 = {0, 1, 2} (1-indexed)
            f.write("3 4\n")       # edge 1 = {2, 3}
            f.write("1 4\n")       # edge 2 = {0, 3}
        hg = HyperGraph.from_hmetis(path)
        assert hg.num_nodes == 4
        assert hg.num_edges == 3

    def test_read_hmetis_with_edge_weights(self, tmp_path):
        path = str(tmp_path / "ew.hgr")
        with open(path, "w") as f:
            f.write("2 3 10\n")     # 2 edges, 3 vertices, fmt=10 (edge weights)
            f.write("5 1 2\n")      # edge weight 5, vertices {0, 1}
            f.write("3 2 3\n")      # edge weight 3, vertices {1, 2}
        hg = HyperGraph.from_hmetis(path)
        np.testing.assert_array_equal(hg.edge_weights, [5, 3])

    def test_read_hmetis_with_both_weights(self, tmp_path):
        path = str(tmp_path / "bw.hgr")
        with open(path, "w") as f:
            f.write("2 3 11\n")     # fmt=11 (both weights)
            f.write("5 1 2\n")      # edge weight 5, vertices {0, 1}
            f.write("3 2 3\n")      # edge weight 3, vertices {1, 2}
            f.write("10\n")         # vertex 0 weight
            f.write("20\n")         # vertex 1 weight
            f.write("30\n")         # vertex 2 weight
        hg = HyperGraph.from_hmetis(path)
        np.testing.assert_array_equal(hg.edge_weights, [5, 3])
        np.testing.assert_array_equal(hg.node_weights, [10, 20, 30])

    def test_comments_skipped(self, tmp_path):
        path = str(tmp_path / "comments.hgr")
        with open(path, "w") as f:
            f.write("c This is a comment\n")
            f.write("1 2\n")       # 1 edge, 2 vertices
            f.write("1 2\n")       # edge 0 = {0, 1}
        hg = HyperGraph.from_hmetis(path)
        assert hg.num_nodes == 2
        assert hg.num_edges == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hypergraph.py -v -k "HMetisIO"`
Expected: FAIL

**Step 3: Implement in `chszlablib/io.py`**

`read_hmetis(path)`:
- Skip comment lines (start with `c` or `%`)
- Parse header: `M N [W]` where M=edges, N=vertices
- W format: 0=no weights, 1=edge weights only, 10=node weights only (hMETIS uses 10 not METIS), 11=both
- Parse M edge lines: if has_edge_weights, first token is weight, rest are 1-indexed vertex IDs
- If has_node_weights (W=10 or W=11), parse N additional lines for vertex weights
- Build HyperGraph via `from_edge_list` with weights

`write_hmetis(hg, path)`:
- Header: `M N [W]`
- Edge lines: `[weight] v1 v2 ...` (1-indexed)
- If node weights present: N additional lines

Add to `HyperGraph`:
- `from_hmetis(path)` — delegates to `read_hmetis`
- `to_hmetis(path)` — delegates to `write_hmetis`

**Step 4: Run tests**

Run: `pytest tests/test_hypergraph.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add chszlablib/io.py chszlablib/hypergraph.py tests/test_hypergraph.py
git commit -m "feat: add hMETIS I/O for HyperGraph (read_hmetis, write_hmetis)"
```

---

### Task 5: HyperGraph.to_graph() clique expansion

**Files:**
- Modify: `chszlablib/hypergraph.py`
- Test: `tests/test_hypergraph.py` (append)

**Step 1: Write failing tests**

```python
from chszlablib import Graph


class TestHyperGraphToGraph:
    def test_clique_expansion_simple(self):
        """Edge {0,1,2} should become triangle 0-1, 0-2, 1-2."""
        hg = HyperGraph.from_edge_list([[0, 1, 2]])
        g = hg.to_graph()
        assert isinstance(g, Graph)
        assert g.num_nodes == 3
        assert g.num_edges == 3  # triangle

    def test_clique_expansion_two_edges(self):
        """Edges {0,1} and {1,2}: graph is path 0-1-2."""
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2]])
        g = hg.to_graph()
        assert g.num_nodes == 3
        assert g.num_edges == 2  # no duplicate for edge 0-1, 1-2

    def test_overlapping_edges_no_duplicate(self):
        """Edges {0,1,2} and {1,2,3}: both produce edge 1-2, but stored only once."""
        hg = HyperGraph.from_edge_list([[0, 1, 2], [1, 2, 3]])
        g = hg.to_graph()
        assert g.num_nodes == 4
        # From {0,1,2}: 0-1, 0-2, 1-2. From {1,2,3}: 1-2 (dup), 1-3, 2-3.
        # Unique: 0-1, 0-2, 1-2, 1-3, 2-3 = 5 edges
        assert g.num_edges == 5

    def test_single_vertex_edge(self):
        """Edge with single vertex produces no graph edges."""
        hg = HyperGraph.from_edge_list([[0], [0, 1]])
        g = hg.to_graph()
        assert g.num_nodes == 2
        assert g.num_edges == 1

    def test_preserves_node_weights(self):
        hg = HyperGraph.from_edge_list([[0, 1]], node_weights=[5, 10])
        g = hg.to_graph()
        np.testing.assert_array_equal(g.node_weights, [5, 10])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hypergraph.py -v -k "ToGraph"`
Expected: FAIL

**Step 3: Implement `to_graph()`**

```python
def to_graph(self) -> "Graph":
    from chszlablib.graph import Graph
    self.finalize()
    g = Graph(self._num_nodes)
    # Set node weights
    for i in range(self._num_nodes):
        if self._node_weights[i] != 1:
            g.set_node_weight(i, int(self._node_weights[i]))
    # Clique expansion: for each hyperedge, add all pairs
    seen = set()
    for e in range(self._num_edges):
        start = int(self._eptr[e])
        end = int(self._eptr[e + 1])
        verts = sorted(self._everts[start:end])
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = int(verts[i]), int(verts[j])
                key = (u, v)
                if key not in seen:
                    seen.add(key)
                    g.add_edge(u, v)
    g.finalize()
    return g
```

**Step 4: Run tests**

Run: `pytest tests/test_hypergraph.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add chszlablib/hypergraph.py tests/test_hypergraph.py
git commit -m "feat: add HyperGraph.to_graph() clique expansion"
```

---

### Task 6: Update public API (__init__.py, describe())

**Files:**
- Modify: `chszlablib/__init__.py`
- Test: `tests/test_hypergraph.py` (append)

**Step 1: Write failing test**

```python
class TestHyperGraphPublicAPI:
    def test_import_from_package(self):
        from chszlablib import HyperGraph, InvalidHyperGraphError, read_hmetis, write_hmetis
        assert HyperGraph is not None
        assert InvalidHyperGraphError is not None

    def test_describe_mentions_hypergraph(self):
        from chszlablib import describe
        text = describe()
        assert "HyperGraph" in text
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hypergraph.py::TestHyperGraphPublicAPI -v`
Expected: FAIL with `ImportError`

**Step 3: Update `__init__.py`**

Add imports:
```python
from chszlablib.exceptions import InvalidHyperGraphError
from chszlablib.hypergraph import HyperGraph
from chszlablib.io import read_hmetis, write_hmetis
```

Add to `__all__`:
```python
"HyperGraph",
"InvalidHyperGraphError",
"read_hmetis",
"write_hmetis",
```

Update `describe()`:
- Add `HYPERGRAPH CONSTRUCTION` section after `GRAPH EXPORT`
- Add `HyperMISResult` to result types (placeholder for now)
- Add `read_hmetis`, `write_hmetis` to I/O section

**Step 4: Run tests**

Run: `pytest tests/test_hypergraph.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add chszlablib/__init__.py tests/test_hypergraph.py
git commit -m "feat: export HyperGraph in public API, update describe()"
```

---

### Task 7: Add HyperMIS git submodule and CMake static library

**Files:**
- Modify: `.gitmodules`
- Modify: `CMakeLists.txt`
- Modify: `build.sh` (submodule already handled generically)

**Step 1: Add submodule**

```bash
cd /home/c_schulz/projects/coding/CHSZLabLib
git submodule add https://github.com/KarlsruheMIS/HyperMIS external_repositories/HyperMIS
```

**Step 2: Add CMake section to `CMakeLists.txt`**

Add after the last static library section (before the pybind11 module definitions):

```cmake
# =============================================================================
# HyperMIS  (Maximum Independent Set on Hypergraphs)
# =============================================================================
set(HYPERMIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HyperMIS)

set(HYPERMIS_SOURCES
    ${HYPERMIS_DIR}/src/config.cpp
    ${HYPERMIS_DIR}/src/hypergraph.cpp
    ${HYPERMIS_DIR}/src/graph.cpp
    ${HYPERMIS_DIR}/src/reductions.cpp
    ${HYPERMIS_DIR}/src/MIS_algorithm.cpp
)

add_library(hypermis_static STATIC ${HYPERMIS_SOURCES})
target_include_directories(hypermis_static PUBLIC
    ${HYPERMIS_DIR}/include
    ${HYPERMIS_DIR}/include/datastructure
    ${HYPERMIS_DIR}/include/meta
    ${HYPERMIS_DIR}/include/utils
)
target_compile_options(hypermis_static PRIVATE -w -O2)

# Optional: Gurobi for ILP solver
find_library(GUROBI_CXX_LIB gurobi_c++ PATHS ENV GUROBI_HOME PATH_SUFFIXES lib)
find_library(GUROBI_LIB NAMES gurobi120 gurobi110 gurobi100 PATHS ENV GUROBI_HOME PATH_SUFFIXES lib)
if(GUROBI_CXX_LIB AND GUROBI_LIB)
    set(HYPERMIS_HAS_GUROBI ON)
    message(STATUS "HyperMIS: Gurobi found, ILP solver will be built")
    add_library(hypermis_ilp_static STATIC ${HYPERMIS_DIR}/src/ILP_solver.cpp)
    target_link_libraries(hypermis_ilp_static PUBLIC hypermis_static ${GUROBI_CXX_LIB} ${GUROBI_LIB})
    target_include_directories(hypermis_ilp_static PUBLIC $ENV{GUROBI_HOME}/include)
else()
    set(HYPERMIS_HAS_GUROBI OFF)
    message(STATUS "HyperMIS: Gurobi not found, ILP solver will NOT be built")
endif()
```

**Step 3: Verify build**

Run: `cd /home/c_schulz/projects/coding/CHSZLabLib && bash build.sh`
Expected: Build succeeds, `hypermis_static` compiles, existing tests still pass

**Step 4: Commit**

```bash
git add .gitmodules external_repositories/HyperMIS CMakeLists.txt
git commit -m "feat: add HyperMIS submodule and CMake static library"
```

---

### Task 8: pybind11 binding for HyperMIS reductions (_hypermis)

**Files:**
- Create: `bindings/hypermis_binding.cpp`
- Modify: `CMakeLists.txt` (add pybind11 module)

**Step 1: Write the binding**

Create `bindings/hypermis_binding.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>

#include "definitions.h"
#include "datastructure/hypergraph.h"
#include "datastructure/fast_set.h"
#include "MIS_algorithm.h"
#include "config.h"
#include "reductions.h"

namespace py = pybind11;

// Build hypergraph from dual CSR arrays (edge-to-vertex representation)
static hypergraph* build_hypergraph_from_csr(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes)
{
    auto eptr_data = eptr.unchecked<1>();
    auto everts_data = everts.unchecked<1>();
    int num_edges = static_cast<int>(eptr.size() - 1);

    // Allocate and zero-init the hypergraph struct
    hypergraph* g = new hypergraph();
    std::memset(g, 0, sizeof(hypergraph));
    g->n = static_cast<NodeID>(num_nodes);
    g->m = static_cast<NodeID>(num_edges);

    // Allocate E arrays (edge -> vertices)
    g->Ed = (NodeID*)calloc(num_edges, sizeof(NodeID));
    g->Ea = (NodeID*)calloc(num_edges, sizeof(NodeID));
    g->E  = (NodeID**)calloc(num_edges, sizeof(NodeID*));

    for (int e = 0; e < num_edges; e++) {
        int start = static_cast<int>(eptr_data(e));
        int end   = static_cast<int>(eptr_data(e + 1));
        int size  = end - start;
        g->Ed[e] = static_cast<NodeID>(size);
        g->Ea[e] = static_cast<NodeID>(size);
        g->E[e]  = (NodeID*)malloc(size * sizeof(NodeID));
        for (int i = 0; i < size; i++) {
            g->E[e][i] = static_cast<NodeID>(everts_data(start + i));
        }
    }

    // Allocate V arrays (vertex -> edges)
    g->Vd = (NodeID*)calloc(num_nodes, sizeof(NodeID));
    g->Va = (NodeID*)calloc(num_nodes, sizeof(NodeID));
    g->V  = (NodeID**)calloc(num_nodes, sizeof(NodeID*));

    // Count degrees
    for (int e = 0; e < num_edges; e++) {
        for (NodeID i = 0; i < g->Ed[e]; i++) {
            g->Vd[g->E[e][i]]++;
        }
    }

    // Allocate and fill V arrays
    for (int v = 0; v < num_nodes; v++) {
        g->Va[v] = g->Vd[v];
        g->V[v] = (NodeID*)malloc(g->Vd[v] * sizeof(NodeID));
        g->Vd[v] = 0;  // Reset to use as insertion counter
    }
    for (int e = 0; e < num_edges; e++) {
        for (NodeID i = 0; i < g->Ed[e]; i++) {
            NodeID v = g->E[e][i];
            g->V[v][g->Vd[v]++] = static_cast<NodeID>(e);
        }
    }

    // N arrays (neighbors) will be built by hypergraph_build_neighbors
    g->Nd = (NodeID*)calloc(num_nodes, sizeof(NodeID));
    g->Na = (NodeID*)calloc(num_nodes, sizeof(NodeID));
    g->N  = (NodeID**)calloc(num_nodes, sizeof(NodeID*));
    for (int v = 0; v < num_nodes; v++) {
        g->N[v] = (NodeID*)malloc(sizeof(NodeID));
        g->Na[v] = 1;
    }

    return g;
}

static std::tuple<int, py::array_t<int32_t>, double>
py_hypermis_reduce(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes,
    double time_limit,
    int seed,
    bool strong_reductions)
{
    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    // Configure globals
    VERBOSE = 0;
    TIME_KERNEL_SECONDS = static_cast<size_t>(time_limit);
    if (strong_reductions) {
        UNCONFINED_REDUCE = 1;
        NUM_REMOVED_EDGES = 1000000;
        CONSTANT_UNCONFINED = 5;
        ITERATIONS_UNCONFINED = 20000;
        EDGE_SIZE = 5000;
        HEURISTIC_RED = 1;
    } else {
        UNCONFINED_REDUCE = 0;
        REDUCE = 1;
        HEURISTIC_RED = 0;
        NUM_REMOVED_EDGES = 20000;
        ITERATIONS_UNCONFINED = 10000;
        CONSTANT_UNCONFINED = 3;
        EDGE_SIZE = 200;
    }

    // Build hypergraph
    hypergraph* g = build_hypergraph_from_csr(eptr, everts, num_nodes);

    // Create algorithm and reduce
    MISH_algorithm* mis_alg = new MISH_algorithm(g);
    hypergraph_build_neighbors(g, &(mis_alg->node_set));

    auto start_time = std::chrono::high_resolution_clock::now();
    mis_alg->reduce_graph();
    auto end_time = std::chrono::high_resolution_clock::now();
    double reduction_time = std::chrono::duration<double>(end_time - start_time).count();

    // Collect IS vertices (those with status == included)
    std::vector<int32_t> is_vertices;
    for (NodeID v = 0; v < g->n; v++) {
        if (mis_alg->status.node_status[v] == MISH_algorithm::IS_status::included) {
            is_vertices.push_back(static_cast<int32_t>(v));
        }
    }

    int offset = static_cast<int>(mis_alg->status.IS_size);

    // Clean up
    hypergraph_free(g);
    delete mis_alg;

    // Restore stdout/stderr
    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Build result array
    py::array_t<int32_t> result(is_vertices.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < is_vertices.size(); i++) {
        r(i) = is_vertices[i];
    }

    return std::make_tuple(offset, result, reduction_time);
}

PYBIND11_MODULE(_hypermis, m) {
    m.doc() = "Python bindings for HyperMIS (hypergraph MIS reductions)";

    m.def("reduce", &py_hypermis_reduce,
          py::arg("eptr"), py::arg("everts"),
          py::arg("num_nodes"),
          py::arg("time_limit"), py::arg("seed"),
          py::arg("strong_reductions"),
          R"doc(
          Run HyperMIS reduction rules on a hypergraph.

          Returns
          -------
          tuple[int, ndarray[int32], float]
              (offset, is_vertices, reduction_time_seconds).
          )doc");
}
```

**Step 2: Add to CMakeLists.txt**

After the hypermis_static section, add:

```cmake
# --- _hypermis (HyperMIS reductions, always built) ---
pybind11_add_module(_hypermis bindings/hypermis_binding.cpp)
target_link_libraries(_hypermis PRIVATE hypermis_static)
install(TARGETS _hypermis DESTINATION chszlablib)

# --- _hypermis_ilp (HyperMIS ILP solver, Gurobi required) ---
if(HYPERMIS_HAS_GUROBI)
    pybind11_add_module(_hypermis_ilp bindings/hypermis_ilp_binding.cpp)
    target_link_libraries(_hypermis_ilp PRIVATE hypermis_ilp_static)
    install(TARGETS _hypermis_ilp DESTINATION chszlablib)
endif()
```

**Step 3: Build**

Run: `bash build.sh`
Expected: `_hypermis.*.so` appears in `chszlablib/`

**Step 4: Commit**

```bash
git add bindings/hypermis_binding.cpp CMakeLists.txt
git commit -m "feat: add pybind11 binding for HyperMIS reductions"
```

---

### Task 9: HyperMISResult and IndependenceProblems.hypermis() Python wrapper

**Files:**
- Modify: `chszlablib/independence.py`
- Modify: `chszlablib/__init__.py`
- Test: `tests/test_hypermis.py` (new)

**Step 1: Write failing tests**

Create `tests/test_hypermis.py`:

```python
"""Tests for HyperMIS (Maximum Independent Set on Hypergraphs)."""

import numpy as np
import pytest

from chszlablib import HyperGraph, IndependenceProblems
from chszlablib.independence import HyperMISResult


def is_valid_hyper_independent_set(hg, vertices):
    """Check that no hyperedge is fully contained in the independent set.

    Strong independence: at most 1 vertex per hyperedge.
    """
    hg.finalize()
    is_set = set(int(v) for v in vertices)
    for e in range(hg.num_edges):
        start = int(hg.eptr[e])
        end = int(hg.eptr[e + 1])
        count = sum(1 for i in range(start, end) if int(hg.everts[i]) in is_set)
        if count > 1:
            return False
    return True


class TestHyperMISResult:
    def test_fields(self):
        r = HyperMISResult(
            size=3, weight=10, vertices=np.array([0, 1, 2]),
            offset=2, reduction_time=0.5,
        )
        assert r.size == 3
        assert r.weight == 10
        assert r.offset == 2
        assert r.reduction_time == 0.5


class TestHyperMIS:
    def test_simple_hypergraph(self):
        """4 vertices, 2 edges: {0,1,2} and {2,3}. MIS can have at most 1 per edge."""
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert isinstance(result, HyperMISResult)
        assert result.size >= 1
        assert is_valid_hyper_independent_set(hg, result.vertices)

    def test_disjoint_edges(self):
        """Disjoint edges: {0,1} and {2,3}. MIS = 2 (one from each)."""
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert result.size == 2
        assert is_valid_hyper_independent_set(hg, result.vertices)

    def test_single_large_edge(self):
        """One edge covering all vertices: MIS = 1."""
        hg = HyperGraph.from_edge_list([[0, 1, 2, 3, 4]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert result.size == 1
        assert is_valid_hyper_independent_set(hg, result.vertices)

    def test_strong_reductions(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3], [3, 4]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0, strong_reductions=True)
        assert result.size >= 1
        assert is_valid_hyper_independent_set(hg, result.vertices)

    def test_weight_sum(self):
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3]], node_weights=[10, 20, 30, 40])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        expected_weight = int(np.sum(hg.node_weights[result.vertices]))
        assert result.weight == expected_weight

    def test_negative_time_limit(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        with pytest.raises(ValueError):
            IndependenceProblems.hypermis(hg, time_limit=-1.0)

    def test_available_methods_includes_hypermis(self):
        methods = IndependenceProblems.available_methods()
        assert "hypermis" in methods


class TestHyperMISILPAvailable:
    def test_attribute_exists(self):
        assert hasattr(IndependenceProblems, "HYPERMIS_ILP_AVAILABLE")
        assert isinstance(IndependenceProblems.HYPERMIS_ILP_AVAILABLE, bool)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hypermis.py -v`
Expected: FAIL with `ImportError`

**Step 3: Implement in `chszlablib/independence.py`**

Add `HyperMISResult` dataclass:
```python
@dataclass
class HyperMISResult:
    """Result of a maximum independent set computation on a hypergraph."""
    size: int
    weight: int
    vertices: np.ndarray
    offset: int
    reduction_time: float
```

Add ILP detection:
```python
try:
    from chszlablib._hypermis_ilp import solve as _hypermis_ilp_solve
    _HYPERMIS_ILP_AVAILABLE = True
except ImportError:
    _HYPERMIS_ILP_AVAILABLE = False
```

Add to `IndependenceProblems`:
```python
HYPERMIS_ILP_AVAILABLE: bool = _HYPERMIS_ILP_AVAILABLE
```

Add `hypermis()` method:
```python
@staticmethod
def hypermis(
    hg: "HyperGraph",
    time_limit: float = 60.0,
    reduction_time_limit: float = 50.0,
    seed: int = 0,
    strong_reductions: bool = False,
) -> HyperMISResult:
    from chszlablib._hypermis import reduce as _reduce

    if time_limit < 0:
        raise ValueError(f"time_limit must be >= 0, got {time_limit}")

    hg.finalize()
    eptr = hg.eptr.astype(np.int64, copy=False)
    everts = hg.everts.astype(np.int32, copy=False)

    offset, is_verts, reduction_time = _reduce(
        eptr, everts, hg.num_nodes,
        reduction_time_limit, seed, strong_reductions,
    )

    # TODO: If ILP available and time remains, solve reduced instance
    # For now: reduction-only result

    weight = int(np.sum(hg.node_weights[is_verts])) if len(is_verts) > 0 else 0
    return HyperMISResult(
        size=len(is_verts),
        weight=weight,
        vertices=is_verts,
        offset=offset,
        reduction_time=reduction_time,
    )
```

Update `available_methods()` to include `"hypermis"`.

Update `__init__.py` to export `HyperMISResult`.

**Step 4: Build and run tests**

Run: `bash build.sh` (to get fresh `_hypermis.so`)
Then: `pytest tests/test_hypermis.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add chszlablib/independence.py chszlablib/__init__.py tests/test_hypermis.py
git commit -m "feat: add IndependenceProblems.hypermis() with HyperMIS reductions"
```

---

### Task 10: (Optional) pybind11 binding for HyperMIS ILP solver (_hypermis_ilp)

**Only if Gurobi is available on the build machine.**

**Files:**
- Create: `bindings/hypermis_ilp_binding.cpp`
- Modify: `chszlablib/independence.py` (use ILP when available)

This task extends `hypermis()` to use the ILP solver on the reduced kernel when Gurobi is available. The binding wraps `ILP_solver()` from `ILP_solver.h`.

The implementation follows the same pattern as `hypermis_binding.cpp` but additionally:
1. Calls `mis_alg->build_reduced_hypergraph()` after reduction
2. Calls `ILP_solver(reduced_hg, remaining_time, ...)` on the reduced instance
3. Combines reduction offset + ILP solution

**Defer this task** until Gurobi is installed on the build machine. The reduction-only mode is functional without it.

---

### Task 11: Integration test and final verification

**Files:**
- Test: `tests/test_integration.py` (append)
- Run all tests

**Step 1: Add integration tests**

Append to `tests/test_integration.py`:

```python
def test_hypergraph_full_pipeline():
    """Build HyperGraph, run HyperMIS, verify result."""
    from chszlablib import HyperGraph, IndependenceProblems
    hg = HyperGraph.from_edge_list([
        [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0],
    ])
    result = IndependenceProblems.hypermis(hg, time_limit=5.0)
    # Verify valid independent set
    is_set = set(int(v) for v in result.vertices)
    for e in range(hg.num_edges):
        start = int(hg.eptr[e])
        end = int(hg.eptr[e + 1])
        count = sum(1 for i in range(start, end) if int(hg.everts[i]) in is_set)
        assert count <= 1, f"Edge {e} has {count} vertices in IS"
    assert result.size == len(result.vertices)


def test_hypergraph_to_graph_then_mis():
    """Convert HyperGraph to Graph via clique expansion, then solve MIS."""
    from chszlablib import HyperGraph, IndependenceProblems
    hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3]])
    g = hg.to_graph()
    result = IndependenceProblems.redumis(g, time_limit=5.0)
    assert result.size >= 1


def test_hypergraph_hmetis_roundtrip(tmp_path):
    """Write and read hMETIS format."""
    from chszlablib import HyperGraph, read_hmetis, write_hmetis
    hg = HyperGraph.from_edge_list([[0, 1, 2], [1, 3]], node_weights=[5, 3, 7, 2])
    path = str(tmp_path / "test.hgr")
    write_hmetis(hg, path)
    hg2 = read_hmetis(path)
    assert hg2.num_nodes == hg.num_nodes
    assert hg2.num_edges == hg.num_edges
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (existing 185 + new hypergraph/hypermis tests)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add HyperGraph + HyperMIS integration tests"
```

---

### Task 12: Update memory and documentation

**Files:**
- Modify: `~/.claude/projects/-home-c-schulz-projects-coding-CHSZLabLib/memory/MEMORY.md`

Update MEMORY.md with:
- `chszlablib/hypergraph.py` — HyperGraph class (dual CSR, builder + batch constructors)
- HyperMIS integration in IndependenceProblems
- hMETIS I/O functions
- `InvalidHyperGraphError` in exception hierarchy

---

## Task Dependency Graph

```
Task 1 (exception)
  └─> Task 2 (HyperGraph core)
        ├─> Task 3 (batch constructors)
        │     └─> Task 4 (hMETIS I/O)
        ├─> Task 5 (clique expansion)
        └─> Task 6 (public API)
              └─> Task 7 (submodule + CMake)
                    └─> Task 8 (pybind11 binding)
                          └─> Task 9 (Python wrapper + tests)
                                ├─> Task 10 (ILP binding, optional)
                                └─> Task 11 (integration tests)
                                      └─> Task 12 (memory update)
```

Tasks 3, 4, 5 can be parallelized after Task 2. Task 6 depends on 2-5. Tasks 7-9 are sequential (build system → binding → wrapper).
