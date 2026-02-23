# chszlablib Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single pip-installable Python package (`chszlablib`) exposing KaHIP, VieCut, VieClus, and CHILS algorithms through a common Graph class, without modifying any library source code.

**Architecture:** External pybind11 modules link against the four C/C++ libraries (built as static libs via a top-level CMakeLists.txt). A pure-Python layer provides a unified `Graph` class backed by CSR arrays and thin wrappers around each binding module.

**Tech Stack:** Python 3.9+, pybind11, CMake, scikit-build-core, numpy, pytest

**Design doc:** `docs/plans/2026-02-23-python-interface-design.md`

---

## Task 1: Project Scaffolding and Build System

**Files:**
- Create: `pyproject.toml`
- Create: `CMakeLists.txt`
- Create: `chszlablib/__init__.py`
- Create: `chszlablib/graph.py` (empty placeholder)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.12"]
build-backend = "scikit_build_core.build"

[project]
name = "chszlablib"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["numpy>=1.20"]

[project.optional-dependencies]
networkx = ["networkx>=2.6"]
scipy = ["scipy>=1.7"]
dev = ["pytest>=7.0"]

[tool.scikit-build]
cmake.build-type = "Release"
wheel.packages = ["chszlablib"]
```

**Step 2: Create top-level `CMakeLists.txt`**

This is the most critical file. It must:
1. Build KaHIP's static library (`kahip_static`) via `add_subdirectory(KaHIP)`
2. Build VieClus sources into a static library (VieClus's CMake creates `vieclus_static` only when building Python module; we need to replicate the target)
3. Handle VieCut as header-only (template-based C++17, no library target - algorithms are headers). The pybind11 module compiles directly against VieCut headers. VieCut requires MPI and TCMalloc as `REQUIRED` - we must set `USE_TCMALLOC=OFF` and handle MPI dependency.
4. Build CHILS from its 4 source files (graph.c, chils.c, chils_internal.c, local_search.c) into a static C library
5. Fetch pybind11 via FetchContent
6. Build 4 pybind11 extension modules: `_kahip`, `_viecut`, `_vieclus`, `_chils`

Key considerations discovered during research:
- **KaHIP** exports `kahip_static` target. Include dirs: `KaHIP/interface/`, `KaHIP/lib/`, etc.
- **VieClus** has `vieclus_static` target but only in Python module build path. Sources live in `VieClus/interface/vieclus_interface.cpp` + object libs `libkaffpa` (59 files from `VieClus/extern/KaHIP/`), `libclustering` (12 files), `libpadygrcl` (6 files), `libeval` (9 files).
- **VieCut** is header-only with C++17 templates. CMake requires MPI(REQUIRED) and TCMalloc(REQUIRED by default). We should NOT `add_subdirectory(VieCut)` - instead, set up include paths and compile `viecut_bind.cpp` directly against headers. Dependencies: `tlx` library (in `VieCut/extlib/tlx`), OpenMP.
- **CHILS** uses Makefile. Sources: `CHILS/src/{graph.c, chils.c, chils_internal.c, local_search.c}`. Flags: `-std=gnu17 -fPIC -fopenmp`. Include: `CHILS/include/`.

```cmake
cmake_minimum_required(VERSION 3.15)
project(chszlablib LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── OpenMP ──
find_package(OpenMP)

# ════════════════════════════════════════════
# 1. KaHIP
# ════════════════════════════════════════════
set(NOMPI ON CACHE BOOL "" FORCE)
set(NONATIVEOPTIMIZATIONS ON CACHE BOOL "" FORCE)
add_subdirectory(KaHIP EXCLUDE_FROM_ALL)

pybind11_add_module(_kahip bindings/kahip_bind.cpp)
target_link_libraries(_kahip PRIVATE kahip_static)
target_include_directories(_kahip PRIVATE KaHIP/interface)
install(TARGETS _kahip DESTINATION chszlablib)

# ════════════════════════════════════════════
# 2. VieClus
# ════════════════════════════════════════════
# VieClus has its own embedded KaHIP copy. Build needed sources directly.
set(VIECLUS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VieClus)
set(VIECLUS_KAHIP ${VIECLUS_DIR}/extern/KaHIP)

# Collect VieClus source files (from VieClus/CMakeLists.txt analysis)
# libkaffpa (59 files), libclustering (12 files), libpadygrcl, libeval + interface
# Rather than listing all 80+ files, use add_subdirectory with NOMPI
set(NOMPI ON CACHE BOOL "" FORCE)
add_subdirectory(VieClus EXCLUDE_FROM_ALL)

pybind11_add_module(_vieclus bindings/vieclus_bind.cpp)
target_link_libraries(_vieclus PRIVATE vieclus_static)
target_include_directories(_vieclus PRIVATE
    ${VIECLUS_DIR}/interface
    ${VIECLUS_KAHIP}/lib
    ${VIECLUS_KAHIP}/lib/io
    ${VIECLUS_KAHIP}/lib/tools
    ${VIECLUS_KAHIP}/lib/partition
    ${VIECLUS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
)
install(TARGETS _vieclus DESTINATION chszlablib)

# ════════════════════════════════════════════
# 3. VieCut (header-only, compile binding directly)
# ════════════════════════════════════════════
set(VIECUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VieCut)

# Build tlx from VieCut's extlib
add_subdirectory(${VIECUT_DIR}/extlib/tlx EXCLUDE_FROM_ALL)

pybind11_add_module(_viecut bindings/viecut_bind.cpp)
target_compile_features(_viecut PRIVATE cxx_std_17)
target_include_directories(_viecut PRIVATE
    ${VIECUT_DIR}/lib
    ${VIECUT_DIR}/app
    ${VIECUT_DIR}/extlib/growt
    ${VIECUT_DIR}/extlib/tlx
)
target_link_libraries(_viecut PRIVATE tlx)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_viecut PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS _viecut DESTINATION chszlablib)

# ════════════════════════════════════════════
# 4. CHILS (C library from source files)
# ════════════════════════════════════════════
set(CHILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CHILS)

add_library(chils_static STATIC
    ${CHILS_DIR}/src/graph.c
    ${CHILS_DIR}/src/chils.c
    ${CHILS_DIR}/src/chils_internal.c
    ${CHILS_DIR}/src/local_search.c
)
target_include_directories(chils_static PUBLIC ${CHILS_DIR}/include)
set_target_properties(chils_static PROPERTIES C_STANDARD 17)
if(OpenMP_C_FOUND)
    target_link_libraries(chils_static PUBLIC OpenMP::OpenMP_C)
endif()

pybind11_add_module(_chils bindings/chils_bind.cpp)
target_link_libraries(_chils PRIVATE chils_static)
target_include_directories(_chils PRIVATE ${CHILS_DIR}/include)
install(TARGETS _chils DESTINATION chszlablib)
```

Note: This CMake is a starting point. Task 6 will iterate on build issues once the bindings exist.

**Step 3: Create placeholder Python files**

`chszlablib/__init__.py`:
```python
from chszlablib.graph import Graph
```

`chszlablib/graph.py`:
```python
# Placeholder - implemented in Task 2
```

`tests/__init__.py`: empty

`tests/conftest.py`:
```python
import pytest
from chszlablib import Graph

@pytest.fixture
def simple_path_graph():
    """Path graph: 0--1--2--3 (all weights 1)."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    return g

@pytest.fixture
def weighted_graph():
    """Small weighted graph for testing."""
    g = Graph(num_nodes=5)
    g.add_edge(0, 1, weight=2)
    g.add_edge(0, 2, weight=3)
    g.add_edge(1, 2, weight=1)
    g.add_edge(1, 3, weight=4)
    g.add_edge(2, 4, weight=5)
    g.add_edge(3, 4, weight=2)
    return g
```

**Step 4: Verify scaffolding**

Run: `ls -la chszlablib/ tests/ pyproject.toml CMakeLists.txt`
Expected: All files present.

**Step 5: Commit**

```bash
git add pyproject.toml CMakeLists.txt chszlablib/ tests/
git commit -m "scaffold: project structure, build system, and test fixtures"
```

---

## Task 2: Graph Class

**Files:**
- Create: `chszlablib/graph.py`
- Create: `tests/test_graph.py`

**Step 1: Write tests for Graph class**

`tests/test_graph.py`:
```python
import numpy as np
import pytest
from chszlablib.graph import Graph


class TestGraphBuilder:
    def test_empty_graph(self):
        g = Graph(num_nodes=3)
        assert g.num_nodes == 3
        assert g.num_edges == 0

    def test_add_edge(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2, weight=5)
        g.finalize()
        assert g.num_nodes == 3
        assert g.num_edges == 2  # undirected count

    def test_add_edge_stores_both_directions(self):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1, weight=3)
        g.finalize()
        # CSR should have 2 entries (0->1 and 1->0)
        assert len(g.adjncy) == 2
        assert len(g.edge_weights) == 2

    def test_node_weights_default_to_one(self):
        g = Graph(num_nodes=3)
        g.finalize()
        np.testing.assert_array_equal(g.node_weights, [1, 1, 1])

    def test_set_node_weight(self):
        g = Graph(num_nodes=3)
        g.set_node_weight(0, 10)
        g.set_node_weight(2, 5)
        g.finalize()
        np.testing.assert_array_equal(g.node_weights, [10, 1, 5])

    def test_edge_weights_default_to_one(self):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1)
        g.finalize()
        np.testing.assert_array_equal(g.edge_weights, [1, 1])

    def test_invalid_node_raises(self):
        g = Graph(num_nodes=2)
        with pytest.raises(ValueError):
            g.add_edge(0, 5)

    def test_duplicate_edge_raises(self):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1)
        with pytest.raises(ValueError):
            g.add_edge(0, 1)

    def test_self_loop_raises(self):
        g = Graph(num_nodes=2)
        with pytest.raises(ValueError):
            g.add_edge(0, 0)

    def test_finalize_idempotent(self):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1)
        g.finalize()
        g.finalize()  # should not raise
        assert g.num_edges == 1


class TestGraphFromCSR:
    def test_from_csr(self):
        xadj = np.array([0, 1, 3, 4], dtype=np.int64)
        adjncy = np.array([1, 0, 2, 1], dtype=np.int32)
        g = Graph.from_csr(xadj, adjncy)
        assert g.num_nodes == 3
        assert g.num_edges == 2

    def test_from_csr_with_weights(self):
        xadj = np.array([0, 1, 2], dtype=np.int64)
        adjncy = np.array([1, 0], dtype=np.int32)
        vwgt = np.array([10, 20], dtype=np.int64)
        ewgt = np.array([5, 5], dtype=np.int64)
        g = Graph.from_csr(xadj, adjncy, node_weights=vwgt, edge_weights=ewgt)
        np.testing.assert_array_equal(g.node_weights, [10, 20])
        np.testing.assert_array_equal(g.edge_weights, [5, 5])


class TestGraphCSRProperties:
    def test_xadj_shape(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.finalize()
        assert len(g.xadj) == 4  # num_nodes + 1

    def test_adjacency_correct(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 2, weight=7)
        g.finalize()
        # Node 0 has neighbor 2, node 1 has no neighbors, node 2 has neighbor 0
        assert g.xadj[0] == 0
        assert g.xadj[1] == 1  # node 0 has 1 edge
        assert g.xadj[2] == 1  # node 1 has 0 edges
        assert g.xadj[3] == 2  # node 2 has 1 edge
        assert g.adjncy[0] == 2
        assert g.adjncy[1] == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL (Graph class not implemented)

**Step 3: Implement `chszlablib/graph.py`**

```python
from __future__ import annotations

import numpy as np


class Graph:
    """Lightweight graph backed by CSR arrays.

    Undirected graphs store each edge in both directions in the CSR.
    """

    def __init__(self, num_nodes: int):
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        self._num_nodes = num_nodes
        self._node_weights = np.ones(num_nodes, dtype=np.int64)
        self._edges: list[tuple[int, int, int]] = []  # (src, dst, weight)
        self._edge_set: set[tuple[int, int]] = set()
        self._finalized = False
        self._xadj: np.ndarray | None = None
        self._adjncy: np.ndarray | None = None
        self._edge_weights_arr: np.ndarray | None = None

    # ── Builder API ──

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add edges after finalization")
        if u < 0 or u >= self._num_nodes or v < 0 or v >= self._num_nodes:
            raise ValueError(
                f"Node index out of range: u={u}, v={v}, num_nodes={self._num_nodes}"
            )
        if u == v:
            raise ValueError("Self-loops are not supported")
        key = (min(u, v), max(u, v))
        if key in self._edge_set:
            raise ValueError(f"Duplicate edge: {u}-{v}")
        self._edge_set.add(key)
        self._edges.append((u, v, weight))

    def set_node_weight(self, node: int, weight: int) -> None:
        if node < 0 or node >= self._num_nodes:
            raise ValueError(f"Node index out of range: {node}")
        self._node_weights[node] = weight

    def finalize(self) -> None:
        if self._finalized:
            return
        # Build CSR from edge list (store both directions)
        degree = np.zeros(self._num_nodes, dtype=np.int64)
        for u, v, _ in self._edges:
            degree[u] += 1
            degree[v] += 1

        xadj = np.zeros(self._num_nodes + 1, dtype=np.int64)
        for i in range(self._num_nodes):
            xadj[i + 1] = xadj[i] + degree[i]

        total_directed = int(xadj[self._num_nodes])
        adjncy = np.zeros(total_directed, dtype=np.int32)
        edge_weights = np.zeros(total_directed, dtype=np.int64)

        offset = xadj[:-1].copy()
        for u, v, w in self._edges:
            adjncy[offset[u]] = v
            edge_weights[offset[u]] = w
            offset[u] += 1
            adjncy[offset[v]] = u
            edge_weights[offset[v]] = w
            offset[v] += 1

        self._xadj = xadj
        self._adjncy = adjncy
        self._edge_weights_arr = edge_weights
        self._finalized = True
        # Free builder state
        self._edges = []
        self._edge_set = set()

    # ── CSR class method ──

    @classmethod
    def from_csr(
        cls,
        xadj: np.ndarray,
        adjncy: np.ndarray,
        node_weights: np.ndarray | None = None,
        edge_weights: np.ndarray | None = None,
    ) -> Graph:
        num_nodes = len(xadj) - 1
        g = cls.__new__(cls)
        g._num_nodes = num_nodes
        g._xadj = np.asarray(xadj, dtype=np.int64)
        g._adjncy = np.asarray(adjncy, dtype=np.int32)
        g._node_weights = (
            np.asarray(node_weights, dtype=np.int64)
            if node_weights is not None
            else np.ones(num_nodes, dtype=np.int64)
        )
        g._edge_weights_arr = (
            np.asarray(edge_weights, dtype=np.int64)
            if edge_weights is not None
            else np.ones(len(adjncy), dtype=np.int64)
        )
        g._finalized = True
        g._edges = []
        g._edge_set = set()
        return g

    # ── Properties ──

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        self.finalize()
        return int(len(self._adjncy) // 2)

    @property
    def xadj(self) -> np.ndarray:
        self.finalize()
        return self._xadj

    @property
    def adjncy(self) -> np.ndarray:
        self.finalize()
        return self._adjncy

    @property
    def node_weights(self) -> np.ndarray:
        return self._node_weights

    @property
    def edge_weights(self) -> np.ndarray:
        self.finalize()
        return self._edge_weights_arr
```

**Step 4: Run tests**

Run: `pytest tests/test_graph.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add chszlablib/graph.py tests/test_graph.py
git commit -m "feat: implement Graph class with CSR backend and builder API"
```

---

## Task 3: METIS I/O

**Files:**
- Create: `chszlablib/io.py`
- Create: `tests/test_io.py`
- Modify: `chszlablib/graph.py` (add `from_metis` and `to_metis`)

**Step 1: Write tests**

`tests/test_io.py`:
```python
import numpy as np
import pytest
from chszlablib.graph import Graph

SAMPLE_METIS = """\
4 4 11
1 2 3 3 5
1 1 3 3 2
1 1 5 2 3 4 7
1 3 7
"""

def test_read_metis(tmp_path):
    p = tmp_path / "test.graph"
    p.write_text(SAMPLE_METIS)
    g = Graph.from_metis(str(p))
    assert g.num_nodes == 4
    assert g.num_edges == 4

def test_roundtrip_metis(tmp_path):
    g = Graph(num_nodes=3)
    g.add_edge(0, 1, weight=2)
    g.add_edge(1, 2, weight=3)
    g.set_node_weight(0, 5)
    p = tmp_path / "out.graph"
    g.to_metis(str(p))
    g2 = Graph.from_metis(str(p))
    assert g2.num_nodes == 3
    assert g2.num_edges == 2
    np.testing.assert_array_equal(g.node_weights, g2.node_weights)

def test_read_unweighted_metis(tmp_path):
    content = "3 2\n2 3\n1 3\n1 2\n"
    p = tmp_path / "uw.graph"
    p.write_text(content)
    g = Graph.from_metis(str(p))
    assert g.num_nodes == 3
    assert g.num_edges == 2

def test_read_metis_with_comments(tmp_path):
    content = "% comment\n2 1\n2\n1\n"
    p = tmp_path / "c.graph"
    p.write_text(content)
    g = Graph.from_metis(str(p))
    assert g.num_nodes == 2
    assert g.num_edges == 1
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_io.py -v`
Expected: FAIL

**Step 3: Implement `chszlablib/io.py`**

Implement METIS parser and writer following the standard METIS format:
- Line 1: `n m [fmt]` where fmt `11` = node+edge weights, `10` = node weights, `1` = edge weights, none = unweighted
- Lines 2..n+1: `[node_weight] neighbor1 [edge_weight1] neighbor2 [edge_weight2] ...`
- Neighbors are 1-indexed in file, 0-indexed internally
- Lines starting with `%` are comments

Add `from_metis(path)` classmethod and `to_metis(path)` method to `Graph`.

**Step 4: Run tests**

Run: `pytest tests/test_io.py -v`
Expected: All PASS

**Step 5: Also run existing graph tests to check nothing is broken**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add chszlablib/io.py tests/test_io.py chszlablib/graph.py
git commit -m "feat: add METIS file I/O for Graph class"
```

---

## Task 4: KaHIP pybind11 Binding

**Files:**
- Create: `bindings/kahip_bind.cpp`
- Create: `chszlablib/partition.py`
- Create: `tests/test_partition.py`

**Step 1: Write the pybind11 binding**

`bindings/kahip_bind.cpp` wraps the C functions from `kaHIP_interface.h`. The key function is `kaffpa()` which takes CSR arrays as `int*` pointers.

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

// KaHIP C interface
extern "C" {
#include "kaHIP_interface.h"
}

namespace py = pybind11;

PYBIND11_MODULE(_kahip, m) {
    m.doc() = "KaHIP graph partitioning bindings";

    m.def("kaffpa", [](
        py::array_t<int> vwgt,
        py::array_t<int> xadj,
        py::array_t<int> adjcwgt,
        py::array_t<int> adjncy,
        int nparts,
        double imbalance,
        bool suppress_output,
        int seed,
        int mode
    ) {
        int n = static_cast<int>(xadj.size() - 1);
        std::vector<int> part(n);
        int edgecut = 0;

        int* vwgt_ptr = vwgt.size() > 0 ? vwgt.mutable_data() : nullptr;
        int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

        kaffpa(&n, vwgt_ptr, xadj.mutable_data(), adjcwgt_ptr,
               adjncy.mutable_data(), &nparts, &imbalance,
               suppress_output, seed, mode, &edgecut, part.data());

        py::array_t<int> part_arr(n);
        std::memcpy(part_arr.mutable_data(), part.data(), n * sizeof(int));
        return py::make_tuple(edgecut, part_arr);
    }, "Run KaHIP graph partitioning",
       py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
       py::arg("adjncy"), py::arg("nparts"), py::arg("imbalance"),
       py::arg("suppress_output"), py::arg("seed"), py::arg("mode"));

    m.def("node_separator", [](
        py::array_t<int> vwgt,
        py::array_t<int> xadj,
        py::array_t<int> adjcwgt,
        py::array_t<int> adjncy,
        int nparts,
        double imbalance,
        bool suppress_output,
        int seed,
        int mode
    ) {
        int n = static_cast<int>(xadj.size() - 1);
        int num_separator_vertices = 0;
        int* separator = nullptr;

        int* vwgt_ptr = vwgt.size() > 0 ? vwgt.mutable_data() : nullptr;
        int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

        node_separator(&n, vwgt_ptr, xadj.mutable_data(), adjcwgt_ptr,
                       adjncy.mutable_data(), &nparts, &imbalance,
                       suppress_output, seed, mode,
                       &num_separator_vertices, &separator);

        py::array_t<int> sep_arr(num_separator_vertices);
        if (num_separator_vertices > 0 && separator) {
            std::memcpy(sep_arr.mutable_data(), separator,
                       num_separator_vertices * sizeof(int));
            delete[] separator;
        }
        return py::make_tuple(num_separator_vertices, sep_arr);
    }, "Compute node separator");

    m.def("node_ordering", [](
        py::array_t<int> xadj,
        py::array_t<int> adjncy,
        bool suppress_output,
        int seed,
        int mode
    ) {
        int n = static_cast<int>(xadj.size() - 1);
        std::vector<int> ordering(n);

        reduced_nd(&n, xadj.mutable_data(), adjncy.mutable_data(),
                   suppress_output, seed, mode, ordering.data());

        py::array_t<int> ord_arr(n);
        std::memcpy(ord_arr.mutable_data(), ordering.data(), n * sizeof(int));
        return ord_arr;
    }, "Compute node ordering via nested dissection");

    // Mode constants
    m.attr("FAST") = 0;
    m.attr("ECO") = 1;
    m.attr("STRONG") = 2;
    m.attr("FASTSOCIAL") = 3;
    m.attr("ECOSOCIAL") = 4;
    m.attr("STRONGSOCIAL") = 5;
}
```

**Step 2: Write Python wrapper**

`chszlablib/partition.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph

_MODE_MAP = {
    "fast": 0,
    "eco": 1,
    "strong": 2,
    "fastsocial": 3,
    "ecosocial": 4,
    "strongsocial": 5,
}


@dataclass
class PartitionResult:
    edgecut: int
    assignment: np.ndarray


@dataclass
class SeparatorResult:
    num_separator_vertices: int
    separator: np.ndarray


@dataclass
class OrderingResult:
    ordering: np.ndarray


def partition(
    g: Graph,
    num_parts: int = 2,
    mode: str = "eco",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
) -> PartitionResult:
    from chszlablib._kahip import kaffpa

    g.finalize()
    mode_int = _MODE_MAP[mode.lower()]
    vwgt = g.node_weights.astype(np.int32, copy=False)
    xadj = g.xadj.astype(np.int32, copy=False)
    adjcwgt = g.edge_weights.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    edgecut, part = kaffpa(
        vwgt, xadj, adjcwgt, adjncy,
        num_parts, imbalance, suppress_output, seed, mode_int,
    )
    return PartitionResult(edgecut=edgecut, assignment=part)


def node_separator(
    g: Graph,
    num_parts: int = 2,
    mode: str = "eco",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
) -> SeparatorResult:
    from chszlablib._kahip import node_separator as _ns

    g.finalize()
    mode_int = _MODE_MAP[mode.lower()]
    vwgt = g.node_weights.astype(np.int32, copy=False)
    xadj = g.xadj.astype(np.int32, copy=False)
    adjcwgt = g.edge_weights.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    num_sep, sep = _ns(
        vwgt, xadj, adjcwgt, adjncy,
        num_parts, imbalance, suppress_output, seed, mode_int,
    )
    return SeparatorResult(num_separator_vertices=num_sep, separator=sep)


def node_ordering(
    g: Graph,
    mode: str = "eco",
    seed: int = 0,
    suppress_output: bool = True,
) -> OrderingResult:
    from chszlablib._kahip import node_ordering as _no

    g.finalize()
    mode_int = _MODE_MAP[mode.lower()]
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    ordering = _no(xadj, adjncy, suppress_output, seed, mode_int)
    return OrderingResult(ordering=ordering)
```

**Step 3: Write tests**

`tests/test_partition.py`:
```python
import numpy as np
import pytest
from chszlablib.graph import Graph
from chszlablib.partition import partition, node_separator, node_ordering


def make_bipartite():
    """Complete bipartite K_{3,3}: clear 2-partition with cut=9."""
    g = Graph(num_nodes=6)
    for i in range(3):
        for j in range(3, 6):
            g.add_edge(i, j)
    return g


def test_partition_bipartite():
    g = make_bipartite()
    result = partition(g, num_parts=2, mode="strong")
    assert result.edgecut >= 0
    assert len(result.assignment) == 6
    assert set(np.unique(result.assignment)) <= {0, 1}


def test_partition_modes():
    g = make_bipartite()
    for mode in ["fast", "eco", "strong"]:
        result = partition(g, num_parts=2, mode=mode)
        assert result.edgecut >= 0


def test_partition_k_parts():
    g = make_bipartite()
    result = partition(g, num_parts=3, mode="fast")
    assert len(np.unique(result.assignment)) <= 3


def test_node_separator_basic():
    g = make_bipartite()
    result = node_separator(g, num_parts=2, mode="strong")
    assert result.num_separator_vertices >= 0


def test_node_ordering_basic():
    g = make_bipartite()
    result = node_ordering(g, mode="fast")
    assert len(result.ordering) == 6
    # Ordering should be a permutation
    assert set(result.ordering) == set(range(6))
```

**Step 4: Build and test**

Run: `pip install -e . && pytest tests/test_partition.py -v`
Expected: All PASS (after resolving any build issues)

**Step 5: Commit**

```bash
git add bindings/kahip_bind.cpp chszlablib/partition.py tests/test_partition.py
git commit -m "feat: add KaHIP partitioning bindings"
```

---

## Task 5: VieCut pybind11 Binding

**Files:**
- Create: `bindings/viecut_bind.cpp`
- Create: `chszlablib/mincut.py`
- Create: `tests/test_mincut.py`

**Step 1: Write the pybind11 binding**

VieCut is header-only C++17 with templates. The binding must:
1. Construct a `graph_access` from CSR arrays using `build_from_metis_weighted()`
2. Create a `mutable_graph` from the `graph_access`
3. Select an algorithm via `selectMincutAlgorithm<GraphPtr>(name)`
4. Call `perform_minimum_cut(G)` to get the cut value
5. Extract per-node partition from `G->getNodeInCut(node)`

Key types from VieCut:
- `typedef uint32_t NodeID`
- `typedef uint64_t EdgeID, EdgeWeight`
- `graph_access` has `build_from_metis_weighted(int n, int* xadj, int* adjncy, int* vwgt, int* adjwgt)` (note: VieCut's version uses `int*` same as KaHIP's)
- `mutable_graph` can be constructed from `graph_access` via its `from_graph_access` static method or by iterating
- Algorithm returns `EdgeWeight` (uint64_t)

`bindings/viecut_bind.cpp`:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <string>

#include "algorithms/global_mincut/algorithms.h"
#include "algorithms/global_mincut/minimum_cut.h"
#include "common/configuration.h"
#include "common/definitions.h"
#include "data_structure/graph_access.h"
#include "data_structure/mutable_graph.h"
#include "tools/random_functions.h"

namespace py = pybind11;

typedef mutable_graph graph_type;
typedef std::shared_ptr<graph_type> GraphPtr;

PYBIND11_MODULE(_viecut, m) {
    m.doc() = "VieCut minimum cut bindings";

    m.def("minimum_cut", [](
        py::array_t<int> xadj_arr,
        py::array_t<int> adjncy_arr,
        py::array_t<int> adjwgt_arr,
        std::string algorithm,
        bool save_cut,
        size_t seed
    ) {
        int n = static_cast<int>(xadj_arr.size() - 1);
        int* xadj = xadj_arr.mutable_data();
        int* adjncy = adjncy_arr.mutable_data();
        int* adjwgt = adjwgt_arr.size() > 0 ? adjwgt_arr.mutable_data() : nullptr;

        // Build graph_access from CSR
        auto ga = std::make_shared<graph_access>();
        if (adjwgt) {
            ga->build_from_metis_weighted(n, xadj, adjncy, nullptr, adjwgt);
        } else {
            ga->build_from_metis(n, xadj, adjncy);
        }

        // Convert to mutable_graph
        auto G = std::make_shared<mutable_graph>();
        G->start_construction(ga->number_of_nodes());
        for (NodeID u = 0; u < ga->number_of_nodes(); ++u) {
            G->new_node();
            for (EdgeID e = ga->get_first_edge(u);
                 e < ga->get_first_invalid_edge(u); ++e) {
                G->new_edge(u, ga->getEdgeTarget(e), ga->getEdgeWeight(e));
            }
        }
        G->finish_construction();

        // Configure
        auto cfg = configuration::getConfig();
        cfg->save_cut = save_cut;
        cfg->seed = seed;
        cfg->algorithm = algorithm;
        random_functions::setSeed(seed);

        // Run algorithm
        auto mc = selectMincutAlgorithm<GraphPtr>(algorithm);
        EdgeWeight cut_value = mc->perform_minimum_cut(G);
        delete mc;

        // Extract partition if save_cut
        py::array_t<int> partition_arr(n);
        if (save_cut) {
            auto* pdata = partition_arr.mutable_data();
            for (NodeID u = 0; u < static_cast<NodeID>(n); ++u) {
                pdata[u] = G->getNodeInCut(u) ? 1 : 0;
            }
        }

        return py::make_tuple(static_cast<uint64_t>(cut_value), partition_arr);
    }, "Compute minimum cut",
       py::arg("xadj"), py::arg("adjncy"), py::arg("adjwgt"),
       py::arg("algorithm") = "vc",
       py::arg("save_cut") = true,
       py::arg("seed") = 0);
}
```

**Step 2: Write Python wrapper**

`chszlablib/mincut.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph

_ALGO_MAP = {
    "viecut": "vc",
    "vc": "vc",
    "noi": "noi",
    "ks": "ks",
    "matula": "matula",
    "pr": "pr",
    "cactus": "cactus",
}


@dataclass
class MincutResult:
    cut_value: int
    partition: np.ndarray


def mincut(
    g: Graph,
    algorithm: str = "viecut",
    seed: int = 0,
) -> MincutResult:
    from chszlablib._viecut import minimum_cut

    g.finalize()
    algo_key = _ALGO_MAP[algorithm.lower()]
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    adjwgt = g.edge_weights.astype(np.int32, copy=False)

    cut_value, partition = minimum_cut(
        xadj, adjncy, adjwgt,
        algorithm=algo_key, save_cut=True, seed=seed,
    )
    return MincutResult(cut_value=int(cut_value), partition=partition)
```

**Step 3: Write tests**

`tests/test_mincut.py`:
```python
import numpy as np
import pytest
from chszlablib.graph import Graph
from chszlablib.mincut import mincut


def make_barbell():
    """Barbell graph: two K3 connected by a single edge. Min cut = 1."""
    g = Graph(num_nodes=6)
    # First triangle
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    # Bridge
    g.add_edge(2, 3)
    # Second triangle
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)
    return g


def test_mincut_barbell():
    g = make_barbell()
    result = mincut(g, algorithm="noi")
    assert result.cut_value == 1
    assert len(result.partition) == 6
    assert set(np.unique(result.partition)) == {0, 1}


def test_mincut_complete_graph():
    """K4 has min cut = 3."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    result = mincut(g, algorithm="noi")
    assert result.cut_value == 3


def test_mincut_weighted():
    """Weighted path: 0--[10]--1--[1]--2. Min cut = 1."""
    g = Graph(num_nodes=3)
    g.add_edge(0, 1, weight=10)
    g.add_edge(1, 2, weight=1)
    result = mincut(g, algorithm="noi")
    assert result.cut_value == 1


def test_mincut_algorithms():
    g = make_barbell()
    for algo in ["noi", "viecut", "ks", "pr"]:
        result = mincut(g, algorithm=algo)
        assert result.cut_value == 1, f"Failed for algorithm {algo}"
```

**Step 4: Build and test**

Run: `pip install -e . && pytest tests/test_mincut.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add bindings/viecut_bind.cpp chszlablib/mincut.py tests/test_mincut.py
git commit -m "feat: add VieCut minimum cut bindings"
```

---

## Task 6: VieClus pybind11 Binding

**Files:**
- Create: `bindings/vieclus_bind.cpp`
- Create: `chszlablib/cluster.py`
- Create: `tests/test_cluster.py`

**Step 1: Write the pybind11 binding**

VieClus has a C interface (`vieclus_interface.h`) with a single function:
```c
void vieclus_clustering(int* n, int* vwgt, int* xadj, int* adjcwgt, int* adjncy,
                        bool suppress_output, int seed, double time_limit,
                        int cluster_upperbound, double* modularity,
                        int* num_clusters, int* clustering);
```

`bindings/vieclus_bind.cpp`:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

extern "C" {
    void vieclus_clustering(int* n, int* vwgt, int* xadj,
                            int* adjcwgt, int* adjncy,
                            bool suppress_output, int seed,
                            double time_limit, int cluster_upperbound,
                            double* modularity, int* num_clusters,
                            int* clustering);
}

namespace py = pybind11;

PYBIND11_MODULE(_vieclus, m) {
    m.doc() = "VieClus graph clustering bindings";

    m.def("cluster", [](
        py::array_t<int> vwgt,
        py::array_t<int> xadj,
        py::array_t<int> adjcwgt,
        py::array_t<int> adjncy,
        bool suppress_output,
        int seed,
        double time_limit,
        int cluster_upperbound
    ) {
        int n = static_cast<int>(xadj.size() - 1);
        std::vector<int> clustering(n);
        double modularity = 0.0;
        int num_clusters = 0;

        int* vwgt_ptr = vwgt.size() > 0 ? vwgt.mutable_data() : nullptr;
        int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

        vieclus_clustering(&n, vwgt_ptr, xadj.mutable_data(), adjcwgt_ptr,
                          adjncy.mutable_data(), suppress_output, seed,
                          time_limit, cluster_upperbound,
                          &modularity, &num_clusters, clustering.data());

        py::array_t<int> clust_arr(n);
        std::memcpy(clust_arr.mutable_data(), clustering.data(), n * sizeof(int));
        return py::make_tuple(modularity, num_clusters, clust_arr);
    }, "Run VieClus graph clustering",
       py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
       py::arg("adjncy"), py::arg("suppress_output"),
       py::arg("seed"), py::arg("time_limit"),
       py::arg("cluster_upperbound"));
}
```

**Step 2: Write Python wrapper**

`chszlablib/cluster.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class ClusterResult:
    modularity: float
    num_clusters: int
    assignment: np.ndarray


def cluster(
    g: Graph,
    time_limit: float = 1.0,
    seed: int = 0,
    cluster_upperbound: int = 0,
    suppress_output: bool = True,
) -> ClusterResult:
    from chszlablib._vieclus import cluster as _cluster

    g.finalize()
    vwgt = g.node_weights.astype(np.int32, copy=False)
    xadj = g.xadj.astype(np.int32, copy=False)
    adjcwgt = g.edge_weights.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    modularity, num_clusters, assignment = _cluster(
        vwgt, xadj, adjcwgt, adjncy,
        suppress_output, seed, time_limit, cluster_upperbound,
    )
    return ClusterResult(
        modularity=modularity,
        num_clusters=num_clusters,
        assignment=assignment,
    )
```

**Step 3: Write tests**

`tests/test_cluster.py`:
```python
import numpy as np
import pytest
from chszlablib.graph import Graph
from chszlablib.cluster import cluster


def make_two_cliques():
    """Two K4 connected by a single edge. Clear community structure."""
    g = Graph(num_nodes=8)
    # Clique 1: nodes 0-3
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    # Clique 2: nodes 4-7
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j)
    # Bridge
    g.add_edge(3, 4)
    return g


def test_cluster_two_cliques():
    g = make_two_cliques()
    result = cluster(g, time_limit=1.0)
    assert result.modularity > 0.0
    assert result.num_clusters >= 2
    assert len(result.assignment) == 8


def test_cluster_modularity_positive():
    g = make_two_cliques()
    result = cluster(g, time_limit=1.0)
    assert result.modularity > 0.3  # two clear communities -> high modularity
```

**Step 4: Build and test**

Run: `pip install -e . && pytest tests/test_cluster.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add bindings/vieclus_bind.cpp chszlablib/cluster.py tests/test_cluster.py
git commit -m "feat: add VieClus clustering bindings"
```

---

## Task 7: CHILS pybind11 Binding

**Files:**
- Create: `bindings/chils_bind.cpp`
- Create: `chszlablib/mwis.py`
- Create: `tests/test_mwis.py`

**Step 1: Write the pybind11 binding**

CHILS has a clean C API (`chils.h`) with an opaque solver pointer. Key flow:
1. `chils_initialize()` -> `void*`
2. `chils_set_graph(solver, n, xadj, adjncy, weights)` - set graph from CSR
3. `chils_run_full(solver, time_limit, n_solutions, seed)` - run MWIS solver
4. `chils_solution_get_weight(solver)` -> weight
5. `chils_solution_get_independent_set(solver)` -> `int*` array of vertex IDs
6. `chils_solution_get_size(solver)` -> number of vertices in set
7. `chils_release(solver)` - cleanup

Note: CHILS uses `long long* xadj` and `int* adjncy` and `long long* weights`.

`bindings/chils_bind.cpp`:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

extern "C" {
#include "chils.h"
}

namespace py = pybind11;

PYBIND11_MODULE(_chils, m) {
    m.doc() = "CHILS maximum weight independent set bindings";

    m.def("mwis", [](
        py::array_t<long long> xadj,
        py::array_t<int> adjncy,
        py::array_t<long long> weights,
        double time_limit,
        int num_concurrent,
        unsigned int seed
    ) {
        int n = static_cast<int>(xadj.size() - 1);

        void* solver = chils_initialize();

        chils_set_graph(solver, n,
                       xadj.data(), adjncy.data(), weights.data());

        chils_run_full(solver, time_limit, num_concurrent, seed);

        long long total_weight = chils_solution_get_weight(solver);
        int set_size = chils_solution_get_size(solver);
        int* is_ptr = chils_solution_get_independent_set(solver);

        py::array_t<int> vertices(set_size);
        if (set_size > 0 && is_ptr) {
            std::memcpy(vertices.mutable_data(), is_ptr, set_size * sizeof(int));
        }

        chils_release(solver);

        return py::make_tuple(total_weight, vertices);
    }, "Solve maximum weight independent set",
       py::arg("xadj"), py::arg("adjncy"), py::arg("weights"),
       py::arg("time_limit"), py::arg("num_concurrent"),
       py::arg("seed"));
}
```

**Step 2: Write Python wrapper**

`chszlablib/mwis.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class MWISResult:
    weight: int
    vertices: np.ndarray


def mwis(
    g: Graph,
    time_limit: float = 10.0,
    num_concurrent: int = 4,
    seed: int = 0,
) -> MWISResult:
    from chszlablib._chils import mwis as _mwis

    g.finalize()

    # CHILS uses long long for xadj and weights, int for adjncy
    xadj = g.xadj.astype(np.int64, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    weights = g.node_weights.astype(np.int64, copy=False)

    total_weight, vertices = _mwis(
        xadj, adjncy, weights,
        time_limit, num_concurrent, seed,
    )
    return MWISResult(weight=int(total_weight), vertices=vertices)
```

**Step 3: Write tests**

`tests/test_mwis.py`:
```python
import numpy as np
import pytest
from chszlablib.graph import Graph
from chszlablib.mwis import mwis


def make_path_weighted():
    """Weighted path: 0(10)--1(1)--2(10). Optimal IS = {0, 2}, weight = 20."""
    g = Graph(num_nodes=3)
    g.set_node_weight(0, 10)
    g.set_node_weight(1, 1)
    g.set_node_weight(2, 10)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def test_mwis_path():
    g = make_path_weighted()
    result = mwis(g, time_limit=1.0, num_concurrent=1)
    assert result.weight == 20
    assert set(result.vertices) == {0, 2}


def test_mwis_independent_set_valid():
    """Verify result is actually an independent set."""
    g = Graph(num_nodes=5)
    for i in range(5):
        g.set_node_weight(i, i + 1)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.finalize()

    result = mwis(g, time_limit=1.0, num_concurrent=1)

    # Check no two vertices in the IS are neighbors
    is_set = set(result.vertices)
    for u in is_set:
        start = g.xadj[u]
        end = g.xadj[u + 1]
        for idx in range(start, end):
            assert g.adjncy[idx] not in is_set, (
                f"Vertices {u} and {g.adjncy[idx]} are both in IS but are neighbors"
            )


def test_mwis_single_node():
    g = Graph(num_nodes=1)
    g.set_node_weight(0, 42)
    result = mwis(g, time_limit=1.0, num_concurrent=1)
    assert result.weight == 42
    assert list(result.vertices) == [0]
```

**Step 4: Build and test**

Run: `pip install -e . && pytest tests/test_mwis.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add bindings/chils_bind.cpp chszlablib/mwis.py tests/test_mwis.py
git commit -m "feat: add CHILS MWIS bindings"
```

---

## Task 8: Unified `__init__.py` and Integration Tests

**Files:**
- Modify: `chszlablib/__init__.py`
- Create: `tests/test_integration.py`

**Step 1: Update `__init__.py`**

```python
from chszlablib.graph import Graph
from chszlablib.partition import partition, node_separator, node_ordering
from chszlablib.mincut import mincut
from chszlablib.cluster import cluster
from chszlablib.mwis import mwis

__all__ = [
    "Graph",
    "partition",
    "node_separator",
    "node_ordering",
    "mincut",
    "cluster",
    "mwis",
]
```

**Step 2: Write integration tests**

`tests/test_integration.py`:
```python
"""Integration tests using example graph files from the libraries."""
import os
import pytest
from chszlablib import Graph, partition, mincut, cluster, mwis

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def kahip_example():
    path = os.path.join(REPO_ROOT, "KaHIP", "examples", "rgg_n_2_15_s0.graph")
    if not os.path.exists(path):
        pytest.skip("KaHIP example graph not found")
    return Graph.from_metis(path)


def test_all_algorithms_on_same_graph():
    """Run all four algorithm families on the same graph."""
    g = Graph(num_nodes=8)
    # Two cliques connected by bridge
    for i in range(4):
        g.set_node_weight(i, 10)
        for j in range(i + 1, 4):
            g.add_edge(i, j, weight=5)
    for i in range(4, 8):
        g.set_node_weight(i, 10)
        for j in range(i + 1, 8):
            g.add_edge(i, j, weight=5)
    g.add_edge(3, 4, weight=1)

    # Partition
    pr = partition(g, num_parts=2, mode="fast")
    assert pr.edgecut >= 0
    assert len(pr.assignment) == 8

    # Min cut
    mc = mincut(g, algorithm="noi")
    assert mc.cut_value == 1

    # Cluster
    cr = cluster(g, time_limit=1.0)
    assert cr.modularity > 0

    # MWIS
    mr = mwis(g, time_limit=1.0, num_concurrent=1)
    assert mr.weight > 0
    # Verify it's valid
    is_set = set(mr.vertices)
    g.finalize()
    for u in is_set:
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            assert g.adjncy[idx] not in is_set


def test_metis_roundtrip_with_algorithms(tmp_path):
    """Build graph, save to METIS, reload, run algorithm."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1)
    g.add_edge(0, 3, weight=1)
    g.add_edge(1, 2, weight=1)
    g.add_edge(1, 3, weight=1)
    g.add_edge(2, 3, weight=1)

    path = str(tmp_path / "k4.graph")
    g.to_metis(path)
    g2 = Graph.from_metis(path)

    result = partition(g2, num_parts=2, mode="fast")
    assert result.edgecut >= 0


def test_large_kahip_example(kahip_example):
    """Partition the KaHIP example graph (32K nodes)."""
    result = partition(kahip_example, num_parts=4, mode="fast")
    assert result.edgecut > 0
    assert len(result.assignment) == kahip_example.num_nodes
```

**Step 3: Build and run all tests**

Run: `pip install -e . && pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add chszlablib/__init__.py tests/test_integration.py
git commit -m "feat: unified API exports and integration tests"
```

---

## Task 9: Build System Debugging and Cross-Platform Fixes

This task is iterative. The CMakeLists.txt from Task 1 is a starting point that will likely need fixes.

**Known issues to watch for:**
1. **KaHIP `add_subdirectory` conflicts**: KaHIP's CMake may define targets that conflict with VieClus's embedded KaHIP copy. May need to rename targets or use `EXCLUDE_FROM_ALL`.
2. **VieClus MPI requirement**: VieClus CMake does `find_package(MPI)` unless `NOMPI` is set. Ensure `NOMPI=ON` is set before `add_subdirectory(VieClus)`.
3. **VieCut MPI requirement**: VieCut's CMake has `find_package(MPI REQUIRED)`. Since we don't `add_subdirectory(VieCut)`, this is avoided. But we need to manually handle the `tlx` dependency.
4. **CHILS C17 on Windows**: MSVC has limited C17 support. May need `gnu17` -> `c17` and workarounds.
5. **OpenMP on macOS**: Apple Clang doesn't ship OpenMP. Need conditional handling.
6. **pybind11 module install paths**: scikit-build-core needs modules installed to `chszlablib/`.

**Step 1: Build on Linux first**

Run: `pip install -e . -v 2>&1 | tee build.log`
Expected: Identify and fix build errors iteratively.

**Step 2: Fix each build error**

Common fixes:
- Add missing include directories
- Resolve symbol conflicts between KaHIP and VieClus's embedded KaHIP
- Handle missing `alloc_traits.h` on some systems (VieCut includes `<ext/alloc_traits.h>`)
- Ensure `-fPIC` on all static libraries

**Step 3: Verify all tests pass**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit fixes**

```bash
git add CMakeLists.txt
git commit -m "fix: resolve build system issues"
```

---

## Task 10: NetworkX / SciPy Interop (Optional)

**Files:**
- Modify: `chszlablib/graph.py` (add to_networkx, from_networkx, to_scipy_sparse)
- Create: `tests/test_interop.py`

Only implement if time permits. These are convenience methods:

```python
# In Graph class:
def to_networkx(self):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(self.num_nodes))
    self.finalize()
    for u in range(self.num_nodes):
        for idx in range(self.xadj[u], self.xadj[u + 1]):
            v = int(self.adjncy[idx])
            if u < v:  # avoid duplicate edges
                G.add_edge(u, v, weight=int(self.edge_weights[idx]))
    return G

@classmethod
def from_networkx(cls, G):
    import networkx as nx
    g = cls(num_nodes=G.number_of_nodes())
    mapping = {n: i for i, n in enumerate(G.nodes())}
    for u, v, data in G.edges(data=True):
        g.add_edge(mapping[u], mapping[v], weight=data.get("weight", 1))
    return g

def to_scipy_sparse(self):
    from scipy.sparse import csr_matrix
    self.finalize()
    return csr_matrix(
        (self.edge_weights, self.adjncy, self.xadj),
        shape=(self.num_nodes, self.num_nodes),
    )
```

**Commit:**
```bash
git add chszlablib/graph.py tests/test_interop.py
git commit -m "feat: add networkx and scipy interop methods"
```

---

## Summary of Task Dependencies

```
Task 1 (Scaffolding)
  └── Task 2 (Graph class)
       └── Task 3 (METIS I/O)
       └── Task 4 (KaHIP binding)  ──┐
       └── Task 5 (VieCut binding) ──┤
       └── Task 6 (VieClus binding)──┤── Task 8 (Unified API + integration)
       └── Task 7 (CHILS binding)  ──┘         │
                                        Task 9 (Build fixes, iterative)
                                               │
                                        Task 10 (Interop, optional)
```

Tasks 4-7 are independent of each other and can be parallelized.
