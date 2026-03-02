# HeiHGM Integration Summary

## Overview

Integrated two hypergraph matching algorithm libraries from [HeiHGM](https://github.com/HeiHGM) into CHSZLabLib's `IndependenceProblems` namespace:

- **HeiHGM/Bmatching** — Static hypergraph b-matching (greedy with 7 orderings, reductions+unfold, ILS)
- **HeiHGM/Streaming** — Streaming hypergraph matching (naive, greedy_set, best_evict, greedy, lenient)

Both original repositories use Bazel with Abseil and Protobuf dependencies. These were replaced with CMake and `std` equivalents.

---

## Files Created

| File | Purpose |
|------|---------|
| `external_repositories/HeiHGM_Bmatching/` | Git submodule — static b-matching algorithms |
| `external_repositories/HeiHGM_Streaming/` | Git submodule — streaming matching algorithms |
| `scripts/patch_heihgm.py` | CMake-time patch script — strips Abseil/Protobuf from both repos |
| `bindings/bmatching_binding.cpp` | pybind11 binding for static b-matching (calls C++ templates directly) |
| `bindings/streaming_bmatching_binding.cpp` | pybind11 binding for streaming matching (reimplements 5 algorithms) |
| `tests/test_bmatching.py` | Test suite — 35 tests covering static, streaming, and capacity APIs |

## Files Modified

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Added HeiHGM section: patch execution, INTERFACE library, two pybind11 modules |
| `chszlablib/hypergraph.py` | Added capacity support: `set_capacity()`, `set_capacities()`, `capacities` property |
| `chszlablib/independence.py` | Added `BMatchingResult`, `bmatching()` method, `StreamingBMatcher` class |
| `chszlablib/__init__.py` | Added exports for `BMatchingResult`, `StreamingBMatcher` |
| `README.md` | Added library table entries, quick start, API reference, citations, authors |

---

## Key Technical Decisions

### 1. Bmatching is Header-Only → INTERFACE Library

The Bmatching `.cc` files contain only `#include "ds/bmatching.h"` — all algorithm logic lives in template headers. CMake uses an `INTERFACE` library (include paths only, no compilation) linked to the pybind11 module.

### 2. Streaming Algorithms Reimplemented in Binding

The original `greedy.cc` (989 lines) is deeply coupled to Protobuf via `AlgorithmConfig`, `StreamingAlgorithmImpl`, and a factory/registration system. Stripping Protobuf from this file would require rewriting most of it.

Instead, the 5 streaming algorithms were reimplemented directly in `streaming_bmatching_binding.cpp` (~200 lines total). Each algorithm's core `Stream()` logic is only 10-20 lines:

- **naive**: Binary matched flag per node
- **greedy**: Phi-based weight tracking with epsilon threshold
- **greedy_set**: Per-vertex shortcut pointers with eviction
- **best_evict**: Tries multiple epsilon values, picks best result
- **lenient**: Half-scaled phi distribution (`weight / |pins|`)

### 3. Abseil Stripping via CMake-Time Patch

Only minimal Abseil usage in core algorithm files:
- `absl::flat_hash_map` in `ils.h` → `std::unordered_map`
- `absl::string_view` in Streaming's `hypergraph.h` → `std::string_view`
- `absl::StripLeadingAsciiWhitespace` → manual `string::erase`

The `scripts/patch_heihgm.py` script runs at CMake configure time (same pattern as existing CluStRE patching). It also replaces `exit()` calls with `throw std::runtime_error()`.

### 4. Capacity Support Added to HyperGraph

The existing `HyperGraph` class was extended with:
- Sparse `_capacity_map` dict during construction
- Dense `_capacities` numpy array after `finalize()`
- Default capacity = 1 for all nodes
- `set_capacity(node, capacity)` for single nodes
- `set_capacities(array)` for bulk assignment
- Validation: capacities must be >= 1, cannot be set after finalization

---

## Build & Verification

| Check | Result |
|-------|--------|
| GCC build (`pip install -e . -v`) | 864/864 targets — clean |
| Clang syntax check (both bindings) | Clean with `-std=c++17 -fsyntax-only` |
| Unit tests (`pytest tests/test_bmatching.py`) | 35/35 passed |
| Full test suite (`pytest tests/`) | 328 passed, 3 skipped, 0 failures |
| Import check | `from chszlablib import IndependenceProblems, StreamingBMatcher, BMatchingResult` — OK |

---

## API Summary

### Static B-Matching

```python
from chszlablib import HyperGraph, IndependenceProblems

hg = HyperGraph.from_edge_list([[0,1], [1,2], [2,3]], num_nodes=4, edge_weights=[5, 3, 7])
result = IndependenceProblems.bmatching(hg, algorithm="greedy_weight_desc")
# result.matched_edges, result.total_weight, result.num_matched
```

**Algorithms:** `greedy_random`, `greedy_weight_desc`, `greedy_weight_asc`, `greedy_degree_asc`, `greedy_degree_desc`, `greedy_weight_degree_ratio_desc`, `greedy_weight_degree_ratio_asc`, `reductions`, `ils`

### Streaming B-Matching

```python
from chszlablib import StreamingBMatcher

sm = StreamingBMatcher(num_nodes=100, algorithm="greedy")
for edge, weight in stream:
    sm.add_edge(edge, weight)
result = sm.finish()
```

**Algorithms:** `naive`, `greedy_set`, `best_evict`, `greedy` (default), `lenient`

---

## Process

1. **Analysis** — Explored both HeiHGM repositories to understand namespaces, APIs, dependencies, and template structure
2. **Submodules** — Added both repos as git submodules
3. **Patch script** — Created `patch_heihgm.py` for Abseil/Protobuf stripping
4. **CMake** — Added INTERFACE library for Bmatching headers + two pybind11 modules
5. **Bindings** — Static binding calls C++ templates directly; streaming binding reimplements algorithms
6. **Python API** — Extended `HyperGraph` with capacities, added `BMatchingResult`, `bmatching()`, `StreamingBMatcher`
7. **Tests** — 35 tests covering all algorithms, capacities, streaming reset, edge cases
8. **Documentation** — Updated README with library table, examples, API reference, citations, authors
