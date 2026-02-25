# HeiCut Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate HeiCut (exact hypergraph minimum cut solver) as `Decomposition.hypermincut()` with 4 methods: kernelizer, submodular, ILP, trimmer.

**Architecture:** HeiCut is added as a git submodule. Mt-KaHyPar (HeiCut's hypergraph data structure library) is built from source as a shared library. HeiCut's solver sources are compiled into a static library. A pybind11 binding bridges Python HyperGraph CSR arrays to Mt-KaHyPar's StaticHypergraph, runs solvers, and returns results. Gurobi is optional — Python gurobipy fallback for the ILP method.

**Tech Stack:** C++17, pybind11, Mt-KaHyPar (TBB, Boost, hwloc), CMake, gurobipy (optional)

**Design doc:** `docs/plans/2026-02-25-heicut-integration-design.md`

---

### Task 1: Add HeiCut submodule and build Mt-KaHyPar

**Files:**
- Modify: `.gitmodules`
- Create: `scripts/build_mtkahypar.sh`
- Modify: `external_repositories/` (add HeiCut submodule)

**Step 1: Add HeiCut as a git submodule**

```bash
cd /home/c_schulz/projects/coding/CHSZLabLib
git submodule add https://github.com/HeiCut/HeiCut.git external_repositories/HeiCut
cd external_repositories/HeiCut
git checkout main
cd ../..
```

**Step 2: Initialize Mt-KaHyPar submodules inside HeiCut**

HeiCut vendors Mt-KaHyPar source at `extern/mt-kahypar/`. We need its submodules:

```bash
cd external_repositories/HeiCut/extern/mt-kahypar
git submodule update --init --recursive
cd ../../../../
```

**Step 3: Build libmtkahypar.so from HeiCut's vendored Mt-KaHyPar**

Create `scripts/build_mtkahypar.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MTK_SRC="${REPO_ROOT}/external_repositories/HeiCut/extern/mt-kahypar"
MTK_BUILD="${MTK_SRC}/build"
MTK_DEST="${REPO_ROOT}/external_repositories/HeiCut/extern/mt-kahypar-library"

if [ -f "${MTK_DEST}/libmtkahypar.so" ]; then
    echo "libmtkahypar.so already exists, skipping build."
    exit 0
fi

echo "Building Mt-KaHyPar from source..."
mkdir -p "${MTK_BUILD}"
cmake -S "${MTK_SRC}" -B "${MTK_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DKAHYPAR_DOWNLOAD_TBB=OFF \
    -DKAHYPAR_DOWNLOAD_BOOST=OFF \
    -DKAHYPAR_ENFORCE_MINIMUM_TBB_VERSION=OFF \
    -DKAHYPAR_PYTHON=OFF \
    -DMT_KAHYPAR_DISABLE_BOOST=OFF

cmake --build "${MTK_BUILD}" --target mtkahypar -j"$(nproc)"

# Find and copy the built .so
SO_FILE=$(find "${MTK_BUILD}" -name "libmtkahypar.so" -type f | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: libmtkahypar.so not found after build"
    exit 1
fi

cp "$SO_FILE" "${MTK_DEST}/libmtkahypar.so"
echo "Installed libmtkahypar.so to ${MTK_DEST}/"
```

```bash
chmod +x scripts/build_mtkahypar.sh
```

**Step 4: Run the build script**

```bash
./scripts/build_mtkahypar.sh
```

Verify: `ls -la external_repositories/HeiCut/extern/mt-kahypar-library/libmtkahypar.so`

**Step 5: Commit**

```bash
git add .gitmodules external_repositories/HeiCut scripts/build_mtkahypar.sh
git commit -m "feat: add HeiCut submodule and Mt-KaHyPar build script"
```

---

### Task 2: CMakeLists.txt — HeiCut static library and _heicut pybind11 module

**Files:**
- Modify: `CMakeLists.txt`

**Context:** HeiCut's sources live under `external_repositories/HeiCut/lib/`. We compile them into `heicut_static`. The sparsehash dependency is for commented-out code, so we can remove that include from `pruner.h` or provide a stub. The Gurobi ILP sources (`ilp.cpp`) are conditionally compiled.

**Step 1: Add HeiCut static library to CMakeLists.txt**

Add after the HyperMIS section (around line 1215):

```cmake
# =============================================================================
# HeiCut  (Exact Minimum Cut for Hypergraphs)
# =============================================================================
set(HEICUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiCut)
set(HEICUT_MTK_LIB_DIR ${HEICUT_DIR}/extern/mt-kahypar-library)

# Find the pre-built libmtkahypar.so
find_library(MTKAHYPAR_LIB
    NAMES mtkahypar
    PATHS ${HEICUT_MTK_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT MTKAHYPAR_LIB)
    message(WARNING "libmtkahypar.so not found in ${HEICUT_MTK_LIB_DIR}. "
                    "Run scripts/build_mtkahypar.sh first. Skipping HeiCut.")
else()
    message(STATUS "Found libmtkahypar.so: ${MTKAHYPAR_LIB}")

    # Find TBB (required by Mt-KaHyPar at runtime)
    find_package(TBB QUIET)

    # HeiCut solver sources (without app/ entry points and without argtable/parse_parameters)
    set(HEICUT_SOURCES
        ${HEICUT_DIR}/lib/solvers/kernelizer.cpp
        ${HEICUT_DIR}/lib/solvers/submodular.cpp
        ${HEICUT_DIR}/lib/coarsening/label_propagation.cpp
        ${HEICUT_DIR}/lib/trimmer/trimmer.cpp
        ${HEICUT_DIR}/lib/decomposition/core_decomposition.cpp
        ${HEICUT_DIR}/lib/utils/random.cpp
    )

    # Optionally include ILP solver if Gurobi is available
    find_library(GUROBI_LIB NAMES gurobi gurobi120 gurobi110 gurobi100 gurobi90
                 PATHS "$ENV{GUROBI_HOME}/lib" NO_DEFAULT_PATH)
    find_library(GUROBI_CXX_LIB NAMES libgurobi_g++8.5.a gurobi_g++5.2
                 PATHS "$ENV{GUROBI_HOME}/lib" NO_DEFAULT_PATH)
    find_path(GUROBI_INCLUDE NAMES gurobi_c++.h
              PATHS "$ENV{GUROBI_HOME}/include" NO_DEFAULT_PATH)

    if(GUROBI_LIB AND GUROBI_CXX_LIB AND GUROBI_INCLUDE)
        message(STATUS "Gurobi found for HeiCut ILP solver")
        list(APPEND HEICUT_SOURCES ${HEICUT_DIR}/lib/solvers/ilp.cpp)
        set(HEICUT_HAS_GUROBI TRUE)
    else()
        message(STATUS "Gurobi not found — HeiCut ILP solver disabled (gurobipy fallback available)")
        set(HEICUT_HAS_GUROBI FALSE)
    endif()

    add_library(heicut_static STATIC ${HEICUT_SOURCES})
    target_include_directories(heicut_static PUBLIC
        ${HEICUT_DIR}
        ${HEICUT_DIR}/extern
        ${HEICUT_DIR}/extern/kahip
        ${HEICUT_DIR}/extern/mt-kahypar
        ${HEICUT_DIR}/extern/mt-kahypar-library
        ${HEICUT_DIR}/extern/kahypar-shared-resources
    )
    target_compile_options(heicut_static PRIVATE -w -O2)
    target_link_libraries(heicut_static PUBLIC ${MTKAHYPAR_LIB})

    if(TBB_FOUND)
        target_link_libraries(heicut_static PUBLIC TBB::tbb)
    endif()

    if(HEICUT_HAS_GUROBI)
        target_include_directories(heicut_static PUBLIC ${GUROBI_INCLUDE})
        target_link_libraries(heicut_static PUBLIC ${GUROBI_CXX_LIB} ${GUROBI_LIB})
        target_compile_definitions(heicut_static PUBLIC HEICUT_HAS_GUROBI)
    endif()

    # --- _heicut pybind11 module ---
    pybind11_add_module(_heicut bindings/heicut_binding.cpp)
    target_link_libraries(_heicut PRIVATE heicut_static)
    if(HEICUT_HAS_GUROBI)
        target_compile_definitions(_heicut PRIVATE HEICUT_HAS_GUROBI)
    endif()
    install(TARGETS _heicut DESTINATION chszlablib)

    # Install libmtkahypar.so alongside the package so it's found at runtime
    install(FILES ${MTKAHYPAR_LIB} DESTINATION chszlablib)
endif()
```

**Step 2: Handle the sparsehash include**

The `#include <sparsehash/dense_hash_map>` in `pruner.h` is for commented-out code. To avoid requiring the sparsehash package, create a stub or patch the include. The simplest approach: install `libsparsehash-dev` as a build dependency (it's header-only and lightweight). Alternatively, wrap the include:

In the binding or via a compile definition, we can just install the package. Note this in the README.

**Step 3: Verify the build compiles**

```bash
source .venv/bin/activate
CXX=g++ pip install -e .
```

Expected: build succeeds (possibly with warnings suppressed by `-w`).

**Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "feat: add HeiCut static library and _heicut pybind11 module to CMake"
```

---

### Task 3: C++ binding — heicut_binding.cpp

**Files:**
- Create: `bindings/heicut_binding.cpp`

**Context:** The binding takes CSR arrays from Python, constructs an Mt-KaHyPar hypergraph via the C API, runs the selected solver, and returns results. Key functions from Mt-KaHyPar C API:
- `mt_kahypar_initialize_thread_pool(num_threads, true)` — must call before any operation
- `mt_kahypar_create_hypergraph(preset, n, m, eptr, everts, edge_weights, node_weights)` — creates hypergraph
- `mt_kahypar::utils::cast<StaticHypergraph>(hg)` — casts to C++ type
- `mt_kahypar_free_hypergraph(hg)` — frees memory

Types from `libmtkahypartypes.h`:
- `mt_kahypar_hypernode_id_t` = `unsigned long int` (same as `size_t` on 64-bit)
- `mt_kahypar_hyperedge_id_t` = `unsigned long int`
- `mt_kahypar_hypernode_weight_t` = `int`
- `mt_kahypar_hyperedge_weight_t` = `int`
- Presets: `DEFAULT` → StaticHypergraph, `HIGHEST_QUALITY` → DynamicHypergraph

**Step 1: Write heicut_binding.cpp**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>

// Mt-KaHyPar C API
#include "libmtkahypar.h"
#include "libmtkahypartypes.h"

// Mt-KaHyPar C++ internals (for cast)
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/datastructures/dynamic_hypergraph.h"

// HeiCut solvers
#include "lib/solvers/kernelizer.h"
#include "lib/solvers/submodular.h"
#include "lib/trimmer/trimmer.h"
#include "lib/orderer/orderer.h"
#include "lib/utils/definitions.h"
#include "lib/utils/random.h"

#ifdef HEICUT_HAS_GUROBI
#include "lib/solvers/ilp.h"
#endif

namespace py = pybind11;
using StaticHypergraph = mt_kahypar::ds::StaticHypergraph;
using DynamicHypergraph = mt_kahypar::ds::DynamicHypergraph;

// Thread pool initialization guard
static bool g_tbb_initialized = false;

static void ensure_tbb_initialized(size_t num_threads) {
    if (!g_tbb_initialized) {
        mt_kahypar_initialize_thread_pool(num_threads, true);
        g_tbb_initialized = true;
    }
}

// Build a StaticHypergraph from CSR arrays via the Mt-KaHyPar C API.
// Returns the opaque wrapper (caller must free with mt_kahypar_free_hypergraph).
static mt_kahypar_hypergraph_t build_static_hypergraph(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes)
{
    auto eptr_buf = eptr.unchecked<1>();
    auto everts_buf = everts.unchecked<1>();
    auto ew_buf = edge_weights.unchecked<1>();

    int num_edges = static_cast<int>(eptr.size() - 1);

    // Mt-KaHyPar expects size_t for hyperedge_indices, unsigned long for hyperedges
    std::vector<size_t> mt_eptr(num_edges + 1);
    for (int i = 0; i <= num_edges; i++)
        mt_eptr[i] = static_cast<size_t>(eptr_buf(i));

    std::vector<mt_kahypar_hyperedge_id_t> mt_everts(everts.size());
    for (ssize_t i = 0; i < everts.size(); i++)
        mt_everts[i] = static_cast<mt_kahypar_hyperedge_id_t>(everts_buf(i));

    std::vector<mt_kahypar_hyperedge_weight_t> mt_ew(num_edges);
    for (int i = 0; i < num_edges; i++)
        mt_ew[i] = static_cast<mt_kahypar_hyperedge_weight_t>(ew_buf(i));

    return mt_kahypar_create_hypergraph(
        DEFAULT,
        static_cast<mt_kahypar_hypernode_id_t>(num_nodes),
        static_cast<mt_kahypar_hyperedge_id_t>(num_edges),
        mt_eptr.data(),
        mt_everts.data(),
        mt_ew.data(),
        nullptr  // node_weights (not used for mincut)
    );
}

// Build a DynamicHypergraph (needed by SubmodularMincut solver).
static mt_kahypar_hypergraph_t build_dynamic_hypergraph(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes)
{
    auto eptr_buf = eptr.unchecked<1>();
    auto everts_buf = everts.unchecked<1>();
    auto ew_buf = edge_weights.unchecked<1>();

    int num_edges = static_cast<int>(eptr.size() - 1);

    std::vector<size_t> mt_eptr(num_edges + 1);
    for (int i = 0; i <= num_edges; i++)
        mt_eptr[i] = static_cast<size_t>(eptr_buf(i));

    std::vector<mt_kahypar_hyperedge_id_t> mt_everts(everts.size());
    for (ssize_t i = 0; i < everts.size(); i++)
        mt_everts[i] = static_cast<mt_kahypar_hyperedge_id_t>(everts_buf(i));

    std::vector<mt_kahypar_hyperedge_weight_t> mt_ew(num_edges);
    for (int i = 0; i < num_edges; i++)
        mt_ew[i] = static_cast<mt_kahypar_hyperedge_weight_t>(ew_buf(i));

    return mt_kahypar_create_hypergraph(
        HIGHEST_QUALITY,  // produces DynamicHypergraph
        static_cast<mt_kahypar_hypernode_id_t>(num_nodes),
        static_cast<mt_kahypar_hyperedge_id_t>(num_edges),
        mt_eptr.data(),
        mt_everts.data(),
        mt_ew.data(),
        nullptr
    );
}

// --- Kernelizer solver ---
static std::tuple<int, double> py_heicut_kernelizer(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes,
    int ordering_type,  // 0=MA, 1=TIGHT, 2=QUEYRANNE
    int lp_iterations,
    int seed,
    int num_threads)
{
    ensure_tbb_initialized(static_cast<size_t>(num_threads));

    // Suppress stdout/stderr from HeiCut
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    int num_edges = static_cast<int>(eptr.size() - 1);

    mt_kahypar_hypergraph_t hg_wrapper = build_static_hypergraph(
        eptr, everts, edge_weights, num_nodes);
    StaticHypergraph &hypergraph = mt_kahypar::utils::cast<StaticHypergraph>(hg_wrapper);

    RandomFunctions::set_seed(seed);

    KernelizerConfig config;
    config.baseSolver = BaseSolver::SUBMODULAR;
    config.orderingType = static_cast<OrderingType>(ordering_type);
    config.orderingMode = OrderingMode::SINGLE;
    config.pruningMode = PruningMode::BEST;
    config.LPNumIterations = static_cast<IterationIndex>(lp_iterations);
    config.LPMode = LabelPropagationMode::CLIQUE_EXPANDED;
    config.numThreads = static_cast<size_t>(num_threads);
    config.verbose = false;
    config.seed = seed;

    Kernelizer kernelizer(config);
    KernelizerResult result = kernelizer.compute_mincut(
        hypergraph,
        static_cast<NodeIndex>(num_nodes),
        static_cast<EdgeIndex>(num_edges));

    mt_kahypar_free_hypergraph(hg_wrapper);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    return std::make_tuple(
        static_cast<int>(result.minEdgeCut),
        result.time);
}

// --- Submodular solver ---
static std::tuple<int, double> py_heicut_submodular(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes,
    int ordering_type,
    int seed,
    int num_threads)
{
    ensure_tbb_initialized(static_cast<size_t>(num_threads));

    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    int num_edges = static_cast<int>(eptr.size() - 1);

    mt_kahypar_hypergraph_t hg_wrapper = build_dynamic_hypergraph(
        eptr, everts, edge_weights, num_nodes);
    DynamicHypergraph &hypergraph = mt_kahypar::utils::cast<DynamicHypergraph>(hg_wrapper);

    RandomFunctions::set_seed(seed);

    // Check if edges have non-unit weights
    auto ew_buf = edge_weights.unchecked<1>();
    bool has_weighted = false;
    for (int i = 0; i < num_edges; i++) {
        if (ew_buf(i) != 1) { has_weighted = true; break; }
    }

    auto start = std::chrono::high_resolution_clock::now();

    SubmodularMincut solver(
        static_cast<NodeIndex>(num_nodes),
        static_cast<EdgeIndex>(num_edges),
        static_cast<OrderingType>(ordering_type),
        OrderingMode::SINGLE,
        has_weighted,
        static_cast<size_t>(num_threads));

    SubmodularMincutResult result = solver.solve(hypergraph);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    mt_kahypar_free_hypergraph(hg_wrapper);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    return std::make_tuple(static_cast<int>(result.minEdgeCut), elapsed);
}

// --- Trimmer solver ---
static std::tuple<int, double> py_heicut_trimmer(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes,
    int ordering_type,
    int seed,
    int num_threads)
{
    ensure_tbb_initialized(static_cast<size_t>(num_threads));

    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    int num_edges = static_cast<int>(eptr.size() - 1);

    // Trimmer works on unweighted hypergraphs — create unit-weight edges
    py::array_t<int32_t> unit_weights(num_edges);
    auto uw = unit_weights.mutable_unchecked<1>();
    for (int i = 0; i < num_edges; i++) uw(i) = 1;

    mt_kahypar_hypergraph_t static_wrapper = build_static_hypergraph(
        eptr, everts, unit_weights, num_nodes);
    StaticHypergraph &static_hg = mt_kahypar::utils::cast<StaticHypergraph>(static_wrapper);

    RandomFunctions::set_seed(seed);

    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Compute vertex ordering for initial upper bound
    Orderer<StaticHypergraph, EdgeWeight> orderer(
        static_cast<NodeIndex>(num_nodes),
        static_cast<EdgeIndex>(num_edges));
    auto ordering_result = orderer.compute_ordering(
        static_hg, static_cast<OrderingType>(ordering_type));
    CutValue minEdgeCut = ordering_result.minEdgeCut;

    // Step 2: Trimmer + exponential search
    Trimmer trimmer(static_hg, static_cast<NodeIndex>(num_nodes),
                    static_cast<EdgeIndex>(num_edges), ordering_result.ordering);

    // Exponential search over k
    CutValue k = 1;
    while (k <= minEdgeCut) {
        // Build k-trimmed certificate
        auto trimmed = trimmer.compute_trimmed_certificate(k);
        // Solve the smaller instance with submodular solver
        if (trimmed.numNodes > 0 && trimmed.numEdges > 0) {
            mt_kahypar_hypergraph_t dyn_wrapper = mt_kahypar_create_hypergraph(
                HIGHEST_QUALITY,
                trimmed.numNodes, trimmed.numEdges,
                trimmed.edgeIndices.data(), trimmed.edges.data(),
                trimmed.edgeWeights.data(), nullptr);
            DynamicHypergraph &dyn_hg = mt_kahypar::utils::cast<DynamicHypergraph>(dyn_wrapper);

            SubmodularMincut sub_solver(
                trimmed.numNodes, trimmed.numEdges,
                static_cast<OrderingType>(ordering_type),
                OrderingMode::SINGLE, false, 1);
            auto sub_result = sub_solver.solve(dyn_hg);

            if (sub_result.minEdgeCut < minEdgeCut)
                minEdgeCut = sub_result.minEdgeCut;

            mt_kahypar_free_hypergraph(dyn_wrapper);

            if (k > minEdgeCut) break;
        }
        k *= 2;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    mt_kahypar_free_hypergraph(static_wrapper);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    return std::make_tuple(static_cast<int>(minEdgeCut), elapsed);
}

#ifdef HEICUT_HAS_GUROBI
// --- ILP solver (only compiled when Gurobi is available) ---
static std::tuple<int, double, bool> py_heicut_ilp(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes,
    double ilp_timeout,
    int seed,
    int num_threads)
{
    ensure_tbb_initialized(static_cast<size_t>(num_threads));

    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    int num_edges = static_cast<int>(eptr.size() - 1);

    mt_kahypar_hypergraph_t hg_wrapper = build_static_hypergraph(
        eptr, everts, edge_weights, num_nodes);
    StaticHypergraph &hypergraph = mt_kahypar::utils::cast<StaticHypergraph>(hg_wrapper);

    auto start = std::chrono::high_resolution_clock::now();

    GRBEnv env;
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();

    ILPMincut solver(
        static_cast<NodeIndex>(num_nodes),
        static_cast<EdgeIndex>(num_edges),
        env, seed, ilp_timeout, ILPMode::BIP,
        static_cast<size_t>(num_threads));

    solver.add_node_variables_and_constraints(hypergraph);
    solver.add_edge_variables_and_constraints(hypergraph);
    solver.set_objective();
    solver.optimize();
    CutValue cut = solver.get_result(hypergraph);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    mt_kahypar_free_hypergraph(hg_wrapper);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    return std::make_tuple(static_cast<int>(cut), elapsed, true);
}
#endif

PYBIND11_MODULE(_heicut, m) {
    m.doc() = "Python bindings for HeiCut (exact hypergraph minimum cut)";

    m.def("kernelizer", &py_heicut_kernelizer,
          py::arg("eptr"), py::arg("everts"),
          py::arg("edge_weights"), py::arg("num_nodes"),
          py::arg("ordering_type"), py::arg("lp_iterations"),
          py::arg("seed"), py::arg("num_threads"),
          "Run HeiCut kernelizer. Returns (cut_value, time).");

    m.def("submodular", &py_heicut_submodular,
          py::arg("eptr"), py::arg("everts"),
          py::arg("edge_weights"), py::arg("num_nodes"),
          py::arg("ordering_type"), py::arg("seed"),
          py::arg("num_threads"),
          "Run submodular vertex-ordering solver. Returns (cut_value, time).");

    m.def("trimmer", &py_heicut_trimmer,
          py::arg("eptr"), py::arg("everts"),
          py::arg("num_nodes"), py::arg("ordering_type"),
          py::arg("seed"), py::arg("num_threads"),
          "Run Chekuri-Xu trimmer. Returns (cut_value, time).");

#ifdef HEICUT_HAS_GUROBI
    m.attr("HAS_GUROBI") = true;
    m.def("ilp", &py_heicut_ilp,
          py::arg("eptr"), py::arg("everts"),
          py::arg("edge_weights"), py::arg("num_nodes"),
          py::arg("ilp_timeout"), py::arg("seed"),
          py::arg("num_threads"),
          "Run Gurobi ILP solver. Returns (cut_value, time, is_optimal).");
#else
    m.attr("HAS_GUROBI") = false;
#endif
}
```

**Important notes for the implementer:**
- The Trimmer integration above is a sketch. The actual Trimmer API may differ — study `lib/trimmer/trimmer.h` and `app/mincut_trimmer.cpp` carefully. The trimmer uses the `Orderer` class to get an MA/Tight ordering, builds backward edges, then does exponential search. Adapt the binding to match the actual C++ API.
- The `KernelizerConfig` fields need to match what's in `lib/parse_parameters/parse_parameters.h`. Check all field names at compile time.
- The `#include` paths depend on the include directories set in CMake. The includes above assume the HeiCut root is an include directory.

**Step 2: Verify it compiles**

```bash
source .venv/bin/activate
CXX=g++ pip install -e .
python -c "import chszlablib._heicut; print('OK')"
```

**Step 3: Commit**

```bash
git add bindings/heicut_binding.cpp
git commit -m "feat: add HeiCut pybind11 binding with kernelizer/submodular/trimmer/ilp"
```

---

### Task 4: Python API — HyperMincutResult and hypermincut()

**Files:**
- Modify: `chszlablib/decomposition.py`
- Modify: `chszlablib/__init__.py`

**Step 1: Add HyperMincutResult dataclass to decomposition.py**

After the existing result dataclasses (around line 130):

```python
@dataclass
class HyperMincutResult:
    """Result of a hypergraph minimum cut computation.

    Attributes
    ----------
    cut_value : int
        The minimum cut value (total weight of cut hyperedges).
    time : float
        Computation time in seconds.
    method : str
        Solver method used.
    """
    cut_value: int
    time: float
    method: str
```

**Step 2: Add HYPERMINCUT_METHODS tuple and ILP availability flag**

In the `Decomposition` class body:

```python
HYPERMINCUT_METHODS: tuple[str, ...] = ("kernelizer", "submodular", "ilp", "trimmer")
"""Valid methods for :meth:`hypermincut`."""

try:
    from chszlablib._heicut import HAS_GUROBI as _HEICUT_HAS_CPP_GUROBI
except ImportError:
    _HEICUT_HAS_CPP_GUROBI = False

try:
    import gurobipy as _gp
    _HEICUT_HAS_GUROBIPY = True
except ImportError:
    _HEICUT_HAS_GUROBIPY = False

HYPERMINCUT_ILP_AVAILABLE: bool = _HEICUT_HAS_CPP_GUROBI or _HEICUT_HAS_GUROBIPY
"""Whether the ILP method is available (C++ Gurobi or gurobipy)."""
```

**Step 3: Add hypermincut() method**

```python
@staticmethod
def hypermincut(
    hg: "HyperGraph",
    method: Literal["kernelizer", "submodular", "ilp", "trimmer"] = "kernelizer",
    time_limit: float = 300.0,
    seed: int = 0,
    num_threads: int = 1,
) -> HyperMincutResult:
    """Compute the exact minimum cut of a hypergraph.

    Finds a partition of the vertex set into two non-empty sets S and
    V\\S that minimizes the total weight of hyperedges intersecting
    both S and V\\S.

    .. math::

        \\min_{\\emptyset \\subset S \\subset V}
        \\sum_{e \\in E : e \\cap S \\neq \\emptyset,\\;
               e \\cap (V \\setminus S) \\neq \\emptyset} w(e)

    Parameters
    ----------
    hg : HyperGraph
        The input hypergraph (must be finalized).
    method : {"kernelizer", "submodular", "ilp", "trimmer"}
        Solver method.

        * ``"kernelizer"`` — Full HeiCut pipeline: provably exact
          reduction rules + submodular base solver. Best general choice.
        * ``"submodular"`` — Standalone vertex-ordering solver
          (Tight ordering). No reductions.
        * ``"ilp"`` — Relaxed-BIP ILP solver. Requires Gurobi
          (C++ or gurobipy). Raises :class:`ImportError` if unavailable.
        * ``"trimmer"`` — Chekuri-Xu trimmer certificates.
          Unweighted hypergraphs only.
    time_limit : float
        Time budget in seconds (used by ILP solver).
    seed : int
        Random seed for reproducibility.
    num_threads : int
        Number of threads (default 1 = sequential).

    Returns
    -------
    HyperMincutResult
        Named result with ``cut_value``, ``time``, and ``method``.

    Raises
    ------
    InvalidModeError
        If ``method`` is not recognized.
    ImportError
        If ``method="ilp"`` and neither Gurobi nor gurobipy is available.

    Examples
    --------
    >>> hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4]])
    >>> r = Decomposition.hypermincut(hg)
    >>> print(r.cut_value)

    >>> r = Decomposition.hypermincut(hg, method="submodular")
    """
    from chszlablib.hypergraph import HyperGraph as HG
    if not isinstance(hg, HG):
        raise TypeError(f"Expected HyperGraph, got {type(hg).__name__}")

    method = method.lower()
    if method not in Decomposition.HYPERMINCUT_METHODS:
        raise InvalidModeError(
            f"Unknown method '{method}'. "
            f"Valid: {Decomposition.HYPERMINCUT_METHODS}"
        )

    hg.finalize()

    if method == "kernelizer":
        from chszlablib import _heicut
        cut_value, elapsed = _heicut.kernelizer(
            hg.eptr, hg.everts, hg.edge_weights.astype(np.int32),
            hg.num_nodes,
            ordering_type=1,  # TIGHT
            lp_iterations=0,
            seed=seed,
            num_threads=num_threads,
        )
        return HyperMincutResult(cut_value=int(cut_value), time=elapsed,
                                 method="kernelizer")

    elif method == "submodular":
        from chszlablib import _heicut
        cut_value, elapsed = _heicut.submodular(
            hg.eptr, hg.everts, hg.edge_weights.astype(np.int32),
            hg.num_nodes,
            ordering_type=1,  # TIGHT
            seed=seed,
            num_threads=num_threads,
        )
        return HyperMincutResult(cut_value=int(cut_value), time=elapsed,
                                 method="submodular")

    elif method == "ilp":
        # Try C++ Gurobi first, fall back to gurobipy
        try:
            from chszlablib import _heicut
            if _heicut.HAS_GUROBI:
                cut_value, elapsed, _ = _heicut.ilp(
                    hg.eptr, hg.everts, hg.edge_weights.astype(np.int32),
                    hg.num_nodes,
                    ilp_timeout=time_limit,
                    seed=seed,
                    num_threads=num_threads,
                )
                return HyperMincutResult(cut_value=int(cut_value),
                                         time=elapsed, method="ilp")
        except ImportError:
            pass

        # Python gurobipy fallback
        try:
            from chszlablib._gurobi_hypermincut_ilp import solve_hypermincut_ilp
        except ImportError:
            raise ImportError(
                "method='ilp' requires Gurobi (C++) or gurobipy. "
                "Install with: pip install gurobipy"
            )
        cut_value, elapsed = solve_hypermincut_ilp(
            hg.eptr, hg.everts, hg.edge_weights,
            hg.num_nodes, time_limit=time_limit,
        )
        return HyperMincutResult(cut_value=int(cut_value), time=elapsed,
                                 method="ilp")

    elif method == "trimmer":
        from chszlablib import _heicut
        cut_value, elapsed = _heicut.trimmer(
            hg.eptr, hg.everts, hg.num_nodes,
            ordering_type=1,  # TIGHT
            seed=seed,
            num_threads=num_threads,
        )
        return HyperMincutResult(cut_value=int(cut_value), time=elapsed,
                                 method="trimmer")
```

**Step 4: Update available_methods()**

In the `available_methods()` dict, add:

```python
"hypermincut": "Exact hypergraph minimum cut (HeiCut)",
```

**Step 5: Update __init__.py**

Add `HyperMincutResult` to the imports from `decomposition`:

```python
from chszlablib.decomposition import (
    ...
    HyperMincutResult,
    ...
)
```

And add to `__all__`:

```python
"HyperMincutResult",
```

**Step 6: Update describe()**

In `__init__.py`, add to the RESULT TYPES section:

```python
"  HyperMincutResult       cut_value, time, method",
```

**Step 7: Commit**

```bash
git add chszlablib/decomposition.py chszlablib/__init__.py
git commit -m "feat: add Decomposition.hypermincut() with 4 solver methods"
```

---

### Task 5: Python ILP fallback — _gurobi_hypermincut_ilp.py

**Files:**
- Create: `chszlablib/_gurobi_hypermincut_ilp.py`

**Context:** This is the gurobipy fallback for `method="ilp"` when C++ Gurobi is not available. The ILP formulation for hypergraph minimum cut:
- Binary variable x_v for each vertex (1 if in set S, 0 if not)
- For each hyperedge e, binary variable y_e (1 if e is cut)
- For each hyperedge e and vertex v in e: y_e >= x_v - x_{v'} for all pairs (linearized: for each v in e, y_e >= x_v - (1 - sum of other x's in e) ... actually the standard formulation is simpler)
- The standard Relaxed-BIP formulation: for each hyperedge e, y_e >= x_v - x_u for all v, u in e. Maximize sum(x_v), constraint sum(y_e * w_e) <= lambda, binary search on lambda. OR: minimize sum(y_e * w_e) with the connectivity constraints.

Actually, the simpler formulation for minimum cut:
- x_v in {0, 1} for each vertex (which side of the cut)
- For each hyperedge e, y_e in {0, 1} (1 if hyperedge is cut)
- For each hyperedge e and each pair v, u in e: y_e >= x_v - x_u and y_e >= x_u - x_v
- Actually for hyperedges (not just 2-edges), the constraint is: y_e >= x_v - x_u for any two vertices v, u in e. But that's O(|e|^2) constraints per edge.
- Simpler: for each hyperedge e, let max_x = max(x_v : v in e), min_x = min(x_v : v in e). Then y_e >= max_x - min_x. Since x is binary, this means: for each v in e, y_e >= x_v - x_u for each u in e.
- Even simpler: y_e = 1 if not all x_v for v in e are equal. This can be modeled as: for each v in e, y_e >= x_v - x_{v'} where v' is some reference vertex in e. Actually: pick any reference vertex v0 in e. Then y_e >= x_v - x_{v0} and y_e >= x_{v0} - x_v for all v in e.

```python
"""Pure Python ILP solver for hypergraph minimum cut using gurobipy."""

from __future__ import annotations

import time
import numpy as np


def solve_hypermincut_ilp(
    eptr: np.ndarray,
    everts: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
    time_limit: float = 300.0,
) -> tuple[int, float]:
    """Solve hypergraph minimum cut via ILP.

    Uses the Relaxed-BIP formulation:
    - Binary x_v for each vertex (partition assignment)
    - Binary y_e for each hyperedge (1 if cut)
    - For each hyperedge e with reference vertex v0:
      y_e >= x_v - x_{v0} and y_e >= x_{v0} - x_v for all v in e
    - Minimize sum(w_e * y_e)
    - At least one vertex in each side: sum(x_v) >= 1 and sum(x_v) <= n-1

    Parameters
    ----------
    eptr : ndarray[int64]
        Edge pointer array.
    everts : ndarray[int32]
        Concatenated vertex lists per edge.
    edge_weights : ndarray
        Edge weights.
    num_nodes : int
        Number of vertices.
    time_limit : float
        Gurobi time limit in seconds.

    Returns
    -------
    tuple[int, float]
        (cut_value, elapsed_seconds)
    """
    import gurobipy as gp
    from gurobipy import GRB

    start = time.monotonic()

    num_edges = len(eptr) - 1
    model = gp.Model("hypermincut")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # Vertex partition variables
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")

    # Hyperedge cut indicator variables
    y = model.addVars(num_edges, vtype=GRB.BINARY, name="y")

    # Constraints: y_e = 1 if hyperedge e is cut
    for e in range(num_edges):
        s = int(eptr[e])
        t = int(eptr[e + 1])
        verts = [int(everts[i]) for i in range(s, t)]
        if len(verts) < 2:
            continue
        v0 = verts[0]
        for v in verts[1:]:
            model.addConstr(y[e] >= x[v] - x[v0])
            model.addConstr(y[e] >= x[v0] - x[v])

    # Non-trivial partition: at least 1 vertex on each side
    model.addConstr(gp.quicksum(x[v] for v in range(num_nodes)) >= 1)
    model.addConstr(gp.quicksum(x[v] for v in range(num_nodes)) <= num_nodes - 1)

    # Objective: minimize total weight of cut hyperedges
    model.setObjective(
        gp.quicksum(int(edge_weights[e]) * y[e] for e in range(num_edges)),
        GRB.MINIMIZE,
    )

    model.optimize()

    cut_value = int(round(model.ObjVal))
    elapsed = time.monotonic() - start

    return cut_value, elapsed
```

**Step 2: Commit**

```bash
git add chszlablib/_gurobi_hypermincut_ilp.py
git commit -m "feat: add Python gurobipy fallback for hypermincut ILP"
```

---

### Task 6: Tests — test_hypermincut.py

**Files:**
- Create: `tests/test_hypermincut.py`

**Context:** Test all 4 methods on small hypergraphs with known minimum cut values.

Known test cases:
- **Two disjoint 2-edges** `[[0,1], [2,3]]`: mincut = 0 (disconnected → no edges to cut if we split {0,1} | {2,3}). Actually wait — minimum cut requires both sides non-empty. For a disconnected hypergraph, mincut = 0 since we can partition along the disconnection.
- **Single 2-edge** `[[0,1]]`: mincut = 1 (must cut the only edge).
- **Path-like** `[[0,1], [1,2], [2,3]]`: mincut = 1 (cut any single edge).
- **Two edges sharing a vertex** `[[0,1,2], [2,3,4]]`: mincut = 1.
- **Weighted edges** `[[0,1], [1,2]]` with weights [3, 5]: mincut = 3.

```python
"""Tests for Decomposition.hypermincut() — HeiCut integration."""

import numpy as np
import pytest

from chszlablib import HyperGraph, Decomposition, HyperMincutResult
from chszlablib.exceptions import InvalidModeError


class TestHypermincutKernelizer:
    """Test method='kernelizer' (default)."""

    def test_single_edge(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        result = Decomposition.hypermincut(hg)
        assert isinstance(result, HyperMincutResult)
        assert result.cut_value == 1
        assert result.method == "kernelizer"
        assert result.time >= 0.0

    def test_two_disjoint_edges(self):
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3]])
        result = Decomposition.hypermincut(hg)
        assert result.cut_value == 0  # disconnected

    def test_path_like(self):
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2], [2, 3]])
        result = Decomposition.hypermincut(hg)
        assert result.cut_value == 1

    def test_two_edges_sharing_vertex(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4]])
        result = Decomposition.hypermincut(hg)
        assert result.cut_value == 1

    def test_weighted_edges(self):
        hg = HyperGraph.from_edge_list(
            [[0, 1], [1, 2]],
            edge_weights=np.array([3, 5]),
        )
        result = Decomposition.hypermincut(hg)
        assert result.cut_value == 3  # cut the lighter edge

    def test_triangle_hyperedge(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2]])
        result = Decomposition.hypermincut(hg)
        assert result.cut_value == 1


class TestHypermincutSubmodular:
    """Test method='submodular'."""

    def test_single_edge(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        result = Decomposition.hypermincut(hg, method="submodular")
        assert result.cut_value == 1
        assert result.method == "submodular"

    def test_path_like(self):
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2], [2, 3]])
        result = Decomposition.hypermincut(hg, method="submodular")
        assert result.cut_value == 1


class TestHypermincutTrimmer:
    """Test method='trimmer' (unweighted only)."""

    def test_single_edge(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        result = Decomposition.hypermincut(hg, method="trimmer")
        assert result.cut_value == 1
        assert result.method == "trimmer"

    def test_path_like(self):
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2], [2, 3]])
        result = Decomposition.hypermincut(hg, method="trimmer")
        assert result.cut_value == 1


class TestHypermincutILP:
    """Test method='ilp'."""

    def test_ilp_without_gurobi(self, monkeypatch):
        """method='ilp' raises ImportError when Gurobi unavailable."""
        # Monkeypatch to hide both C++ and Python Gurobi
        import chszlablib.decomposition as mod
        monkeypatch.setattr(mod.Decomposition, "_HEICUT_HAS_CPP_GUROBI", False)
        monkeypatch.setattr(mod.Decomposition, "_HEICUT_HAS_GUROBIPY", False)

        hg = HyperGraph.from_edge_list([[0, 1]])
        with pytest.raises(ImportError, match="gurobipy"):
            Decomposition.hypermincut(hg, method="ilp")

    @pytest.mark.skipif(
        not Decomposition.HYPERMINCUT_ILP_AVAILABLE,
        reason="Gurobi not available"
    )
    def test_ilp_small(self):
        hg = HyperGraph.from_edge_list([[0, 1], [1, 2], [2, 3]])
        result = Decomposition.hypermincut(hg, method="ilp", time_limit=30.0)
        assert result.cut_value == 1
        assert result.method == "ilp"


class TestHypermincutValidation:
    """Test error handling."""

    def test_invalid_method(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        with pytest.raises(InvalidModeError, match="Unknown method"):
            Decomposition.hypermincut(hg, method="bogus")


class TestHypermincutIntrospection:
    """Verify hypermincut appears in available_methods."""

    def test_in_available_methods(self):
        methods = Decomposition.available_methods()
        assert "hypermincut" in methods

    def test_methods_tuple(self):
        assert "kernelizer" in Decomposition.HYPERMINCUT_METHODS
        assert "submodular" in Decomposition.HYPERMINCUT_METHODS
        assert "ilp" in Decomposition.HYPERMINCUT_METHODS
        assert "trimmer" in Decomposition.HYPERMINCUT_METHODS
```

**Step 2: Run tests**

```bash
pytest tests/test_hypermincut.py -v
```

Expected: all tests pass (ILP tests may skip without Gurobi).

**Step 3: Commit**

```bash
git add tests/test_hypermincut.py
git commit -m "feat: add hypermincut tests for all solver methods"
```

---

### Task 7: README and final verification

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Add HeiCut to the integrated libraries table, the API reference section, and the one-liner recipes. Add system dependencies note for TBB/Boost/hwloc.

Key sections to update:
- Integrated libraries table: add HeiCut row
- System dependencies: note TBB, Boost, hwloc requirement
- API reference: add `Decomposition.hypermincut()` section with parameter table and examples
- One-liner recipes: add hypermincut example

**Step 2: Run full test suite**

```bash
source .venv/bin/activate
CXX=g++ pip install -e .
pytest tests/ -v
```

Expected: all tests pass (including new hypermincut tests).

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add HeiCut to README with hypermincut API docs"
```

**Step 4: Push**

```bash
git push
```
