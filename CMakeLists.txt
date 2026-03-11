cmake_minimum_required(VERSION 3.15)
project(chszlablib LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# On macOS with Homebrew libomp, CMake's FindOpenMP may not populate
# OpenMP_CXX_INCLUDE_DIRS.  Propagate it when available.
if(OpenMP_CXX_FOUND AND OpenMP_CXX_INCLUDE_DIRS)
    include_directories(SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
endif()

# Several bundled C++ libraries use APIs removed in C++17 (std::random_shuffle,
# std::binary_function, std::unary_function). On Apple Clang with libc++ these
# are hard errors. Re-enable them globally.
add_compile_definitions(
    _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE
    _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION
)
# KaMIS's bundled KaHIP uses `using namespace __gnu_cxx;` which doesn't exist
# on Apple Clang. Force-include a header that provides an empty namespace.
# Also provide shims for GCC-only headers like <parallel/numeric>.
if(NOT MSVC)
    add_compile_options("SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compat_gnu_cxx.h")
endif()
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# MSVC compatibility: enable alternative operator tokens and suppress warnings.
if(MSVC)
    add_compile_options(/FIiso646.h /W0 /permissive-)
endif()

# =============================================================================
# MQLib  (for fpt-max-cut; must come BEFORE the global -include cstdint
# workaround which breaks MQLib's older C++ code)
# =============================================================================
set(FPT_MAXCUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/fpt-max-cut)
set(MQLIB_DIR ${FPT_MAXCUT_DIR}/solvers/MQLib)
file(GLOB_RECURSE MQLIB_SOURCES ${MQLIB_DIR}/src/*.cpp)
list(FILTER MQLIB_SOURCES EXCLUDE REGEX "main\\.cpp$")
add_library(mqlib_static STATIC ${MQLIB_SOURCES})
target_include_directories(mqlib_static PUBLIC ${MQLIB_DIR}/include)
if(MSVC)
    target_compile_options(mqlib_static PRIVATE /FIcstdint /FIlimits)
else()
    target_compile_options(mqlib_static PRIVATE -w -O2
        "SHELL:-include cstdint"
        "SHELL:-include limits"
    )
endif()

# Fix missing <cstdint> includes in older tlx/VieCut headers on newer compilers
if(NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-include;cstdint>")
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/FIcstdint>")
endif()

# =============================================================================
# KaHIP  (provides kahip_static target)
# =============================================================================
set(NOMPI ON CACHE BOOL "" FORCE)
set(NONATIVEOPTIMIZATIONS ON CACHE BOOL "" FORCE)
set(PARHIP OFF CACHE BOOL "" FORCE)
set(BUILDPYTHONMODULE OFF CACHE BOOL "" FORCE)
add_subdirectory(external_repositories/KaHIP EXCLUDE_FROM_ALL)

# =============================================================================
# VieClus  (built manually to avoid target-name conflicts with KaHIP)
# =============================================================================
# VieClus carries its own copy of KaHIP sources under extern/KaHIP.
# We compile everything into a single static library.

set(VIECLUS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/VieClus)

set(VIECLUS_KAFFPA_SOURCES
    ${VIECLUS_DIR}/extern/KaHIP/lib/data_structure/graph_hierarchy.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/algorithms/strongly_connected_components.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/algorithms/topological_sort.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/algorithms/push_relabel.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/io/graph_io.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/tools/quality_metrics.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/tools/random_functions.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/tools/graph_extractor.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/tools/misc.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/tools/partition_snapshooter.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/graph_partitioner.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/coarsening.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/contraction.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/matching.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/random_matching.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/path.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/clustering/node_ordering.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/bipartition.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_node_separator.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/uncoarsening.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/area_bfs.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/node_separators/greedy_ns_local_search.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/node_separators/fm_ns_local_search.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/node_separators/localized_fm_ns_local_search.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/algorithms/cycle_search.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    ${VIECLUS_DIR}/extern/argtable3-3.0.3/argtable3.c
)

set(VIECLUS_CLUSTERING_SOURCES
    ${VIECLUS_DIR}/lib/parallel_mh_clustering/parallel_mh_async_clustering.cpp
    ${VIECLUS_DIR}/lib/parallel_mh_clustering/population_clustering.cpp
    ${VIECLUS_DIR}/lib/parallel_mh_clustering/exchange/exchanger_clustering.cpp
    ${VIECLUS_DIR}/lib/tools/graph_communication.cpp
    ${VIECLUS_DIR}/lib/clustering/louvainmethod.cpp
    ${VIECLUS_DIR}/lib/clustering/labelpropagation.cpp
    ${VIECLUS_DIR}/lib/clustering/neighborhood.cpp
    ${VIECLUS_DIR}/lib/clustering/coarsening/coarsening.cpp
    ${VIECLUS_DIR}/lib/clustering/coarsening/contractor.cpp
    ${VIECLUS_DIR}/lib/logging/bexception.cpp
    ${VIECLUS_DIR}/lib/tools/modularitymetric.cpp
    ${VIECLUS_DIR}/lib/tools/mpi_tools.cpp
)

add_library(vieclus_static STATIC
    ${VIECLUS_DIR}/interface/vieclus_interface.cpp
    ${VIECLUS_KAFFPA_SOURCES}
    ${VIECLUS_CLUSTERING_SOURCES}
)

target_include_directories(vieclus_static PUBLIC
    ${VIECLUS_DIR}
    ${VIECLUS_DIR}/extern/KaHIP/app
    ${VIECLUS_DIR}/extern/argtable3-3.0.3
    ${VIECLUS_DIR}/extern/KaHIP/lib
    ${VIECLUS_DIR}/extern/KaHIP/lib/io
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition
    ${VIECLUS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${VIECLUS_DIR}/extern/KaHIP/lib/tools
    ${VIECLUS_DIR}/lib
    ${VIECLUS_DIR}/app
    ${VIECLUS_DIR}/interface
)

target_compile_definitions(vieclus_static PRIVATE USE_PSEUDO_MPI)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vieclus_static PUBLIC OpenMP::OpenMP_CXX)
else()
    # VieClus provides an OpenMP stub via its KaHIP misc/ directory
    target_include_directories(vieclus_static PRIVATE ${VIECLUS_DIR}/extern/KaHIP/misc)
endif()

# =============================================================================
# VieCut  (header-only library; only build tlx from its extlib)
# =============================================================================
set(VIECUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/VieCut)

# tlx is the only compiled dependency VieCut needs
add_subdirectory(${VIECUT_DIR}/extlib/tlx EXCLUDE_FROM_ALL)

# =============================================================================
# CHILS  (plain C library -- no existing CMake)
# =============================================================================
set(CHILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/CHILS)

add_library(chils_static STATIC
    ${CHILS_DIR}/src/graph.c
    ${CHILS_DIR}/src/chils.c
    ${CHILS_DIR}/src/chils_internal.c
    ${CHILS_DIR}/src/local_search.c
)

target_include_directories(chils_static PUBLIC ${CHILS_DIR}/include)
set_target_properties(chils_static PROPERTIES C_STANDARD 17)

if(OpenMP_C_FOUND AND NOT MSVC)
    # MSVC's OpenMP 2.0 rejects C99-style for-loop init inside #pragma omp for
    # (error C3015).  Skip OpenMP on MSVC; CHILS will run single-threaded.
    target_link_libraries(chils_static PUBLIC OpenMP::OpenMP_C)
endif()

# =============================================================================
# red2pack  (Maximum 2-Packing Set — reduce-and-transform)
# =============================================================================
set(R2P_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/red2pack)

# ---------------------------------------------------------------------------
# Patch chils_utils.h: red2pack expects the OLD CHILS C API (graph/chils/
# local_search structs with chils_init, chils_run, graph_free, etc.).
# Our CHILS provides a new opaque API (chils_initialize, chils_set_graph,
# chils_run_full, chils_release, etc.).  Replace the entire file.
# ---------------------------------------------------------------------------
set(_R2P_CHILS_UTILS "${R2P_DIR}/lib/red2pack-chils/algorithms/chils_utils.h")
file(READ "${_R2P_CHILS_UTILS}" _r2p_chils_src)
string(FIND "${_r2p_chils_src}" "CHSZLABLIB_CHILS_PATCHED" _r2p_chils_patched)
if(_r2p_chils_patched EQUAL -1)
    file(WRITE "${_R2P_CHILS_UTILS}" [=[
// CHSZLABLIB_CHILS_PATCHED — rewritten for new CHILS opaque API
#pragma once

#include <omp.h>
#include <red2pack/data_structure/m2s_graph_access.h>
#include <red2pack/data_structure/sized_vector.h>

#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace chils {

extern "C" {
#include <chils.h>
}

// Local graph struct replacing the old CHILS C graph type
struct graph {
    int n;
    int *V;   // xadj (size n+1)
    int *E;   // adjncy
    long long *W;  // vertex weights
};

inline void graph_free(graph *g) {
    if (!g) return;
    free(g->V);
    free(g->E);
    free(g->W);
    free(g);
}

struct chils_config {
    bool verbose = true, warm_start = false;
    int run_chils = 1, max_queue = 32;
    double timeout = 3600, step = 5;
    long long il = std::numeric_limits<long long>::max();
    int path_offset = 0, path_end = 0;
    unsigned seed = 0;
};

inline graph *build_graph(red2pack::m2s_graph_access &input_graph) {
    using namespace red2pack;
    int N = static_cast<int>(input_graph.number_of_nodes());
    long long M = static_cast<long long>(input_graph.number_of_edges() + input_graph.number_of_links());

    int *V = (int *)malloc(sizeof(int) * (N + 1));
    int *E = (int *)malloc(sizeof(int) * M);
    long long *W = (long long *)malloc(sizeof(long long) * N);

    int ei = 0;
    forall_nodes(input_graph, u) {
        *(W + u) = static_cast<long long>(input_graph.getNodeWeight(u));
        V[u] = ei;
        forall_out_edges(input_graph, e, u) { E[ei++] = input_graph.getEdgeTarget(e); } endfor
        forall_out_links(input_graph, e, u) { E[ei++] = input_graph.getLinkTarget(e); } endfor
        std::sort(E + V[u], E + ei);
    } endfor
    V[N] = ei;

    graph *g = (graph *)malloc(sizeof(graph));
    *g = graph{N, V, E, W};
    return g;
}

inline graph *build_induced_subgraph(red2pack::m2s_graph_access &input_graph,
                                     const std::vector<bool> &mask,
                                     const red2pack::sized_vector<red2pack::NodeID> &reverse_mapping,
                                     const red2pack::sized_vector<red2pack::NodeID> &mapping) {
    using namespace red2pack;
    int N = static_cast<int>(reverse_mapping.size());
    long long M = 0;
    for (NodeID u : reverse_mapping) {
        forall_out_edges(input_graph, e, u) { if (mask[input_graph.getEdgeTarget(e)]) M++; } endfor
        forall_out_links(input_graph, e, u) { if (mask[input_graph.getLinkTarget(e)]) M++; } endfor
    }

    int *V = (int *)malloc(sizeof(int) * (N + 1));
    int *E = (int *)malloc(sizeof(int) * M);
    long long *W = (long long *)malloc(sizeof(long long) * N);

    int ei = 0;
    for (NodeID i = 0; i < reverse_mapping.size(); ++i) {
        NodeID u = reverse_mapping[i];
        *(W + i) = static_cast<long long>(input_graph.getNodeWeight(u));
        V[i] = ei;
        forall_out_edges(input_graph, e, u) {
            if (mask[input_graph.getEdgeTarget(e)]) E[ei++] = mapping[input_graph.getEdgeTarget(e)];
        } endfor
        forall_out_links(input_graph, e, u) {
            if (mask[input_graph.getLinkTarget(e)]) E[ei++] = mapping[input_graph.getLinkTarget(e)];
        } endfor
        std::sort(E + V[i], E + ei);
    }
    V[N] = ei;

    graph *g = (graph *)malloc(sizeof(graph));
    *g = graph{N, V, E, W};
    return g;
}

inline long long mwis_validate(graph *g, int *independent_set) {
    long long cost = 0;
    for (int u = 0; u < g->n; u++) {
        if (!independent_set[u]) continue;
        cost += g->W[u];
        for (int i = g->V[u]; i < g->V[u + 1]; i++)
            if (independent_set[g->E[i] ]) return -1;
    }
    return cost;
}

inline void run_chils_impl(graph *g, std::vector<bool> &solution_indicator,
                           double &time_best, const chils_config &cfg) {
    int n = g->n;

    // Build long long xadj for new CHILS API
    std::vector<long long> xadj_ll(n + 1);
    for (int i = 0; i <= n; i++) xadj_ll[i] = g->V[i];

    void *solver = chils_initialize();
    chils_set_graph(solver, n, xadj_ll.data(), g->E, g->W);

    if (cfg.run_chils > 1) {
        chils_run_full(solver, cfg.timeout, cfg.run_chils, cfg.seed);
    } else {
        chils_run_local_search_only(solver, cfg.timeout, cfg.seed);
    }

    time_best = chils_solution_get_time(solver);

    for (int i = 0; i < n; i++)
        solution_indicator[i] = (chils_solution_get_vertex_configuration(solver, i) != 0);

    chils_release(solver);
}

inline void run_chils(red2pack::m2s_graph_access &input_graph,
                      std::vector<bool> &solution, double &time_best,
                      const chils_config &cfg) {
    graph *g = build_graph(input_graph);
    run_chils_impl(g, solution, time_best, cfg);
    graph_free(g);
}

}  // namespace chils
]=])
endif()

# ---------------------------------------------------------------------------
# Patch m2s_log.h: suppress scoped_timer output (no-op in library mode)
# ---------------------------------------------------------------------------
set(_R2P_SCOPED_TIMER "${R2P_DIR}/lib/red2pack/tools/scoped_timer.h")
file(READ "${_R2P_SCOPED_TIMER}" _r2p_scoped_src)
string(FIND "${_r2p_scoped_src}" "CHSZLABLIB_SCOPED_PATCHED" _r2p_scoped_patched)
if(_r2p_scoped_patched EQUAL -1)
    file(WRITE "${_R2P_SCOPED_TIMER}" [=[
// CHSZLABLIB_SCOPED_PATCHED — no-op scoped_timer for library use
#pragma once
#include <string>
#include "red2pack/tools/timer.h"
#include "red2pack/tools/m2s_log.h"
#define RED2PACK_SCOPED_TIMER(name)
namespace red2pack {
class scoped_timer {
public:
    explicit scoped_timer(std::string) {}
    ~scoped_timer() {}
};
}
]=])
endif()

# ---------------------------------------------------------------------------
# red2pack_static — fat static library
# ---------------------------------------------------------------------------

# Core library sources
set(R2P_CORE_SOURCES
    ${R2P_DIR}/lib/red2pack/algorithms/kernel/reduce_algorithm.cpp
    ${R2P_DIR}/lib/red2pack/algorithms/kernel/weighted_reduce_algorithm.cpp
    ${R2P_DIR}/lib/red2pack/algorithms/kernel/reductions.cpp
    ${R2P_DIR}/lib/red2pack/algorithms/kernel/weighted_reductions.cpp
    ${R2P_DIR}/lib/red2pack/algorithms/kernel/heuristic_reduction.cpp
    ${R2P_DIR}/lib/red2pack/algorithms/kernel/reduce_and_transform.cpp
    ${R2P_DIR}/lib/red2pack/algorithms/solver_scheme.cpp
    ${R2P_DIR}/lib/red2pack/tools/m2s_log.cpp
    ${R2P_DIR}/lib/red2pack/tools/m2s_graph_io.cpp
)

# Solver sources (each sub-library)
set(R2P_KAMIS_WMIS_SOURCES
    ${R2P_DIR}/lib/red2pack-kamis-wmis/algorithms/branch_and_reduce.cpp
    ${R2P_DIR}/lib/red2pack-kamis-wmis/algorithms/weighted_rnt_exact.cpp
)

set(R2P_CHILS_SOURCES
    ${R2P_DIR}/lib/red2pack-chils/algorithms/weighted_rnt_chils.cpp
    ${R2P_DIR}/lib/red2pack-chils/algorithms/drp.cpp
)

set(R2P_HTWIS_HILS_SOURCES
    ${R2P_DIR}/lib/red2pack-htwis-hils/algorithms/weighted_rnt_htwis.cpp
    ${R2P_DIR}/lib/red2pack-htwis-hils/algorithms/weighted_rnt_hils.cpp
)

set(R2P_MMWIS_SOURCES
    ${R2P_DIR}/lib/red2pack-mmwis/algorithms/weighted_rnt_mmwis.cpp
)

set(R2P_ONLINEMIS_SOURCES
    ${R2P_DIR}/lib/red2pack-onlinemis/algorithms/heuristic.cpp
)

# Bundled extern libraries
set(R2P_HTWIS_SOURCES
    ${R2P_DIR}/extern/htwis/Graph.cpp
)

set(R2P_HILS_SOURCES
    ${R2P_DIR}/extern/hils/src/ArgPack.cpp
    ${R2P_DIR}/extern/hils/src/Graph.cpp
    ${R2P_DIR}/extern/hils/src/Solution.cpp
    ${R2P_DIR}/extern/hils/src/bossa_timer.cpp
)

set(R2P_BUNDLED_ONLINEMIS_SOURCES
    ${R2P_DIR}/extern/onlinemis/data_structure/candidate_list.cpp
    ${R2P_DIR}/extern/onlinemis/data_structure/mis_permutation_online.cpp
    ${R2P_DIR}/extern/onlinemis/data_structure/priority_queues/bucket_array.cpp
    ${R2P_DIR}/extern/onlinemis/ils/local_search_online.cpp
    ${R2P_DIR}/extern/onlinemis/ils/online_ils.cpp
    ${R2P_DIR}/extern/onlinemis/initial_mis/greedy_mis_online.cpp
    ${R2P_DIR}/extern/onlinemis/tools/mis_log.cpp
)

add_library(red2pack_static STATIC
    ${R2P_CORE_SOURCES}
    ${R2P_KAMIS_WMIS_SOURCES}
    ${R2P_CHILS_SOURCES}
    ${R2P_HTWIS_HILS_SOURCES}
    ${R2P_ONLINEMIS_SOURCES}
    ${R2P_HTWIS_SOURCES}
    ${R2P_HILS_SOURCES}
    ${R2P_BUNDLED_ONLINEMIS_SOURCES}
    $<TARGET_OBJECTS:r2p_mmwis_obj>
)

target_include_directories(red2pack_static PUBLIC
    # Core red2pack headers — only the top-level lib dir.
    # Do NOT add subdirs (lib/red2pack, lib/red2pack/tools, etc.) because they
    # shadow KaMIS wmis headers (timer.h, data_structure/sized_vector.h).
    # Red2pack sources use "red2pack/..." prefixed includes that resolve via lib/.
    ${R2P_DIR}/lib
    # Sub-library headers (no onlinemis/mmwis at top level — they have conflicting headers)
    ${R2P_DIR}/lib/red2pack-kamis-wmis
    ${R2P_DIR}/lib/red2pack-kamis-wmis/algorithms
    ${R2P_DIR}/lib/red2pack-chils
    ${R2P_DIR}/lib/red2pack-chils/algorithms
    ${R2P_DIR}/lib/red2pack-htwis-hils
    ${R2P_DIR}/lib/red2pack-htwis-hils/algorithms
    # Bundled extern headers (no onlinemis top-level — has conflicting mis_config.h)
    ${R2P_DIR}/extern
    ${R2P_DIR}/extern/hils/src
    ${R2P_DIR}/extern/onlinemis/data_structure
    ${R2P_DIR}/extern/onlinemis/data_structure/priority_queues
    ${R2P_DIR}/extern/onlinemis/ils
    ${R2P_DIR}/extern/onlinemis/initial_mis
    ${R2P_DIR}/extern/onlinemis/tools
    # App headers (m2s_configuration_*.h)
    ${R2P_DIR}/app
)

# onlinemis top-level dir has mis_config.h that conflicts with KaMIS wmis.
# Add it ONLY to the onlinemis sources via per-file compile options.
set_source_files_properties(
    ${R2P_ONLINEMIS_SOURCES} ${R2P_BUNDLED_ONLINEMIS_SOURCES}
    PROPERTIES COMPILE_OPTIONS
    "-I${R2P_DIR}/extern/onlinemis;-I${R2P_DIR}/lib/red2pack-onlinemis;-I${R2P_DIR}/lib/red2pack-onlinemis/algorithms"
)

# MMWIS solver needs mmwis-namespaced branch_and_reduce_algorithm.h (conflicts with wmis).
# Build as separate OBJECT library with mmwis include dirs taking priority.
add_library(r2p_mmwis_obj OBJECT ${R2P_MMWIS_SOURCES})
set(R2P_KAMIS_MMWIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/KaMIS/mmwis)
target_include_directories(r2p_mmwis_obj BEFORE PRIVATE
    ${R2P_KAMIS_MMWIS_DIR}/lib/mis/kernel
    ${R2P_KAMIS_MMWIS_DIR}/lib/mis/evolutionary
    ${R2P_KAMIS_MMWIS_DIR}/lib/tools
    ${R2P_KAMIS_MMWIS_DIR}/app
    ${R2P_DIR}/lib/red2pack-mmwis
    ${R2P_DIR}/lib/red2pack-mmwis/algorithms
)
target_include_directories(r2p_mmwis_obj PRIVATE
    ${R2P_DIR}/lib
    ${R2P_DIR}/app
)
target_link_libraries(r2p_mmwis_obj PRIVATE kamis_mmwis_static kamis_wmis_static chils_static)
target_compile_options(r2p_mmwis_obj PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)
target_compile_definitions(r2p_mmwis_obj PRIVATE NDEBUG)
if(OpenMP_CXX_FOUND)
    target_link_libraries(r2p_mmwis_obj PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link KaMIS wmis/mmwis (for branch_and_reduce_algorithm, MISConfig, graph_access, etc.)
target_link_libraries(red2pack_static PUBLIC
    kamis_wmis_static
    kamis_mmwis_static
    chils_static
)

target_compile_options(red2pack_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)
target_compile_definitions(red2pack_static PRIVATE NDEBUG)

if(OpenMP_CXX_FOUND)
    target_link_libraries(red2pack_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# LearnAndReduce  (GNN-guided MWIS kernelization)
# =============================================================================
set(LR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/LearnAndReduce)
set(LR_REDUCER ${LR_DIR}/solver/reducer)

add_library(learnandreduce_static STATIC
    # lib/
    ${LR_DIR}/lib/io/graph_io.cpp
    ${LR_DIR}/lib/algorithms/push_relabel.cpp
    ${LR_DIR}/lib/algorithms/strongly_connected_components.cpp
    ${LR_DIR}/lib/data_structure/operation_log.cpp
    ${LR_DIR}/lib/data_structure/priority_queues/bucket_array.cpp
    ${LR_DIR}/lib/data_structure/candidate_list.cpp
    ${LR_DIR}/lib/data_structure/mis_permutation.cpp
    ${LR_DIR}/lib/tools/random_functions.cpp
    ${LR_DIR}/lib/tools/log.cpp
    ${LR_DIR}/lib/tools/graph_extractor.cpp
    ${LR_DIR}/lib/basics/cout_handler.cpp
    # solver/tiny_solver/
    ${LR_DIR}/solver/tiny_solver/tiny_solver.c
    # solver/reducer/lib/
    ${LR_REDUCER}/lib/reduce_algorithm.cpp
    ${LR_REDUCER}/lib/bounds.cpp
    # solver/reducer/reductions/ (core + GNN)
    ${LR_REDUCER}/reductions/general_reduction.cpp
    ${LR_REDUCER}/reductions/LRConv.cpp
    ${LR_REDUCER}/reductions/lr_gcn.cpp
    # simple reductions
    ${LR_REDUCER}/reductions/simple_reduction_rules/fold1_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/fold2_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/neighborhood_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/clique_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/clique_neighborhood_fast_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/twin_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/extended_twin_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/domination_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/extended_domination_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/extended_domination_reverse_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/single_edge_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/extended_single_edge_reduction.cpp
    ${LR_REDUCER}/reductions/simple_reduction_rules/funnel_reduction.cpp
    # expensive reductions
    ${LR_REDUCER}/reductions/expensive_reduction_rules/clique_neighborhood_reduction.cpp
    ${LR_REDUCER}/reductions/expensive_reduction_rules/critical_set_reduction.cpp
    ${LR_REDUCER}/reductions/expensive_reduction_rules/cut_vertex_reduction.cpp
    ${LR_REDUCER}/reductions/expensive_reduction_rules/generalized_fold_reduction.cpp
    ${LR_REDUCER}/reductions/expensive_reduction_rules/heavy_set_reduction.cpp
    ${LR_REDUCER}/reductions/expensive_reduction_rules/heavy_set3_reduction.cpp
    # struction reductions
    ${LR_REDUCER}/reductions/struction_reduction_rules/mwis_finder.cpp
    ${LR_REDUCER}/reductions/struction_reduction_rules/original_struction.cpp
    ${LR_REDUCER}/reductions/struction_reduction_rules/extended_struction.cpp
    ${LR_REDUCER}/reductions/struction_reduction_rules/key_functions.cpp
    ${LR_REDUCER}/reductions/struction_reduction_rules/struction_reductions.cpp
    # CSR reductions
    ${LR_REDUCER}/reductions/csr_reductions/src/reductions.c
    ${LR_REDUCER}/reductions/csr_reductions/unconfined_csr_reduction.cpp
)

target_include_directories(learnandreduce_static PUBLIC
    ${LR_DIR}/lib
    ${LR_DIR}/lib/io
    ${LR_DIR}/lib/config
    ${LR_DIR}/lib/tools
    ${LR_DIR}/lib/algorithms
    ${LR_DIR}/lib/basics
    ${LR_DIR}/lib/data_structure
    ${LR_DIR}/lib/data_structure/priority_queues
    ${LR_DIR}/solver/tiny_solver
    ${LR_REDUCER}/lib
    ${LR_REDUCER}/lib/bounds
    ${LR_REDUCER}/reductions
    ${LR_REDUCER}/reductions/simple_reduction_rules
    ${LR_REDUCER}/reductions/expensive_reduction_rules
    ${LR_REDUCER}/reductions/heuristic_reduction_rules
    ${LR_REDUCER}/reductions/struction_reduction_rules
    ${LR_REDUCER}/reductions/csr_reductions
    ${LR_REDUCER}/reductions/csr_reductions/include
)

target_compile_definitions(learnandreduce_static PUBLIC
    MODE64BITNODEWEIGHTS
    LR_MODELS_FOLDER_OVERRIDE
)
target_compile_options(learnandreduce_static PRIVATE -w)

if(OpenMP_CXX_FOUND)
    target_link_libraries(learnandreduce_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# Patch general_reduction.h: override get_models_folder_path() to read global
set(_LR_GENRED_H "${LR_REDUCER}/reductions/general_reduction.h")
file(READ "${_LR_GENRED_H}" _LR_GENRED_SRC)
string(FIND "${_LR_GENRED_SRC}" "LR_MODELS_FOLDER_OVERRIDE" _LR_ALREADY_PATCHED)
if(_LR_ALREADY_PATCHED EQUAL -1)
    string(REPLACE
        "std::string get_models_folder_path()\n    {\n        std::string current_file = __FILE__;"
        "std::string get_models_folder_path()\n    {\n#ifdef LR_MODELS_FOLDER_OVERRIDE\n        extern std::string g_lr_models_path;\n        if (!g_lr_models_path.empty()) return g_lr_models_path;\n#endif\n        std::string current_file = __FILE__;"
        _LR_GENRED_SRC "${_LR_GENRED_SRC}")
    file(WRITE "${_LR_GENRED_H}" "${_LR_GENRED_SRC}")
endif()

# Patch general_reduction.h: clang fix — extract br_alg->status.n access from
# template into a non-inline helper so the incomplete type is not dereferenced
# in the header (GCC defers this, clang does not).
file(READ "${_LR_GENRED_H}" _LR_GENRED_SRC2)
string(FIND "${_LR_GENRED_SRC2}" "_lr_get_status_n" _LR_CLANG_PATCHED)
if(_LR_CLANG_PATCHED EQUAL -1)
    string(REPLACE
        "template <typename F>\n    inline void for_each_changed_vertex(reduce_algorithm *br_alg, F f)\n    {\n        for (size_t v_idx = 0; v_idx < marker.current_size(); v_idx++)\n        {\n            NodeID v = marker.current_vertex(v_idx);\n\n            if (v < br_alg->status.n && !is_reduced(v, br_alg))"
        "static size_t _lr_get_status_n(reduce_algorithm *br_alg);\n\n    template <typename F>\n    inline void for_each_changed_vertex(reduce_algorithm *br_alg, F f)\n    {\n        for (size_t v_idx = 0; v_idx < marker.current_size(); v_idx++)\n        {\n            NodeID v = marker.current_vertex(v_idx);\n\n            if (v < _lr_get_status_n(br_alg) && !is_reduced(v, br_alg))"
        _LR_GENRED_SRC2 "${_LR_GENRED_SRC2}")
    file(WRITE "${_LR_GENRED_H}" "${_LR_GENRED_SRC2}")
    # Add helper implementation to general_reduction.cpp
    set(_LR_GENRED_CPP "${LR_REDUCER}/reductions/general_reduction.cpp")
    file(READ "${_LR_GENRED_CPP}" _LR_GENRED_CPP_SRC)
    string(FIND "${_LR_GENRED_CPP_SRC}" "_lr_get_status_n" _LR_CPP_PATCHED)
    if(_LR_CPP_PATCHED EQUAL -1)
        string(APPEND _LR_GENRED_CPP_SRC "\nsize_t general_reduction::_lr_get_status_n(reduce_algorithm *br_alg) { return br_alg->status.n; }\n")
        file(WRITE "${_LR_GENRED_CPP}" "${_LR_GENRED_CPP_SRC}")
    endif()
endif()

# Patch LRConv.cpp: remove #include <immintrin.h> (x86-only, unused on ARM)
set(_LR_LRCONV_CPP "${LR_REDUCER}/reductions/LRConv.cpp")
file(READ "${_LR_LRCONV_CPP}" _LR_LRCONV_SRC)
string(FIND "${_LR_LRCONV_SRC}" "#include <immintrin.h>" _LR_LRCONV_HAS_IMMINTRIN)
if(NOT _LR_LRCONV_HAS_IMMINTRIN EQUAL -1)
    string(REPLACE "#include <immintrin.h>" "// #include <immintrin.h>  // removed: x86-only, unused" _LR_LRCONV_SRC "${_LR_LRCONV_SRC}")
    file(WRITE "${_LR_LRCONV_CPP}" "${_LR_LRCONV_SRC}")
endif()

# Patch lr_gcn.cpp: replace AVX2 intrinsics with portable scalar code
set(_LR_GCN_CPP "${LR_REDUCER}/reductions/lr_gcn.cpp")
file(READ "${_LR_GCN_CPP}" _LR_GCN_SRC)
string(FIND "${_LR_GCN_SRC}" "immintrin.h" _LR_GCN_HAS_AVX)
if(NOT _LR_GCN_HAS_AVX EQUAL -1)
    # Replace the entire file with a portable scalar version.  The API
    # (lr_gcn_parse, compute_features, lr_gcn_predict, etc.) is kept
    # identical so that all call-sites continue to compile unchanged.
    file(WRITE "${_LR_GCN_CPP}" [[
#include "lr_gcn.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>

const int param_offsets[4] = {0, 272, 800, 1456};
const int layer_inputs[4] = {8, 16, 40, 16};
const int layer_outputs[4] = {16, 16, 16, 1};
const int max_input = 16;
const int cat_input = 40;
const int layer_output = 16;
const int num_features = 8;
const int node_weight = 0;
const int diff_neighborhood_weight = 1;
const int min_neighborhood_weight = 2;
const int max_neighborhood_weight = 3;
const int node_degree = 4;
const int avg_neighborhood_degree = 5;
const int min_neighborhood_degree = 6;
const int max_neighborhood_degree = 7;

int lr_gcn_parse(FILE *f, float **paramv) {
    int paramc;
    int np = fscanf(f, "%d\n", &paramc);
    *paramv = (float *)malloc(sizeof(float) * paramc);
    for (int i = 0; i < paramc; i++)
        np = fscanf(f, "%f\n", (*paramv) + i);
    (void)np;
    return paramc;
}

void compute_features(graph_access &g, float *x) {
    for (int u = 0; u < (int)g.number_of_nodes(); u++) {
        float *ux = x + u * num_features;
        for (int i = 0; i < num_features; i++) ux[i] = 0.0f;
        ux[node_weight] = g.getNodeWeight(u);
        ux[diff_neighborhood_weight] = g.getNodeWeight(u);
        ux[node_degree] = g.getNodeDegree(u);
        float scale = 1.0f / ux[node_degree];
        for (int i = g.get_first_edge(u); i != (int)g.get_first_invalid_edge(u); i++) {
            int v = g.getEdgeTarget(i);
            float d = g.getNodeDegree(v);
            float w = g.getNodeWeight(v);
            ux[diff_neighborhood_weight] -= w;
            if (w < ux[min_neighborhood_weight] || ux[min_neighborhood_weight] == 0.0f)
                ux[min_neighborhood_weight] = w;
            if (w > ux[max_neighborhood_weight])
                ux[max_neighborhood_weight] = w;
            ux[avg_neighborhood_degree] += d * scale;
            if (d < ux[min_neighborhood_degree] || ux[min_neighborhood_degree] == 0.0f)
                ux[min_neighborhood_degree] = d;
            if (d > ux[max_neighborhood_degree])
                ux[max_neighborhood_degree] = d;
        }
    }
}

static void matrix_multiplication_scalar(const float *A, const float *B, float *C,
                                          const float *b, int n, int k) {
    const int m = layer_output;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) C[i * m + j] = b[j];
        for (int p = 0; p < k; p++) {
            float a = A[i * k + p];
            for (int j = 0; j < m; j++)
                C[i * m + j] += a * B[p * m + j];
        }
    }
}

static void lr_conv_combined_scalar(graph_access &g, int dim, float *in, float *out,
                                     const float *p, const float *b) {
    int N = g.number_of_nodes();
    float *in_buff = (float *)malloc(sizeof(float) * dim * 2);
    for (int u = 0; u < N; u++) {
        float *x = in + u * dim;
        float *y = out + u * layer_output;
        for (int j = 0; j < layer_output; j++) y[j] = 0.0f;
        for (int j = 0; j < dim; j++) in_buff[j] = x[j];
        float scale = 1.0f / (float)g.getNodeDegree(u);
        for (int i = g.get_first_edge(u); i != (int)g.get_first_invalid_edge(u); i++) {
            int v = g.getEdgeTarget(i);
            float *xv = in + v * dim;
            for (int j = 0; j < dim; j++) in_buff[dim + j] = xv[j];
            for (int j = 0; j < layer_output; j++) {
                float val = b[j];
                for (int k = 0; k < dim * 2; k++)
                    val += in_buff[k] * p[k * layer_output + j];
                y[j] += val * scale;
            }
        }
    }
    free(in_buff);
}

void ReLU(float *x, int n) {
    for (int u = 0; u < n; u++)
        for (int i = 0; i < layer_output; i++)
            x[u * layer_output + i] = x[u * layer_output + i] > 0.0f ? x[u * layer_output + i] : 0.0f;
}

void Sigmoid(float *x, int n) {
    for (int u = 0; u < n; u++)
        for (int i = 0; i < layer_output; i++)
            x[u * layer_output + i] = 1.0f / (1.0f + expf(-x[u * layer_output + i]));
}

void prep_params(const float *p, int n, int m, float *out_p, float *out_b) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < layer_output; j++)
            out_p[i * layer_output + j] = 0.0f;
    for (int j = 0; j < layer_output; j++) out_b[j] = 0.0f;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            out_p[i * layer_output + j] = p[j * n + i];
    for (int j = 0; j < m; j++)
        out_b[j] = p[n * m + j];
}

float *lr_gcn_predict(graph_access &g, std::string path) {
    float *params = NULL;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return NULL;
    lr_gcn_parse(f, &params);
    fclose(f);
    int N = g.number_of_nodes();
    float *x = (float *)calloc(max_input * N, sizeof(float));
    float *y = (float *)calloc(max_input * N, sizeof(float));
    float *cat = (float *)calloc(cat_input * N, sizeof(float));
    float *p = (float *)calloc(cat_input * layer_output, sizeof(float));
    float *b = (float *)calloc(layer_output, sizeof(float));
    compute_features(g, x);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < layer_inputs[0]; j++)
            cat[i * cat_input + j] = x[i * layer_inputs[0] + j];
    prep_params(params + param_offsets[0], layer_inputs[0] * 2, layer_outputs[0], p, b);
    lr_conv_combined_scalar(g, layer_inputs[0], x, y, p, b);
    ReLU(y, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < layer_output; j++)
            cat[i * cat_input + layer_inputs[0] + j] = y[i * layer_output + j];
    prep_params(params + param_offsets[1], layer_inputs[1] * 2, layer_outputs[1], p, b);
    lr_conv_combined_scalar(g, layer_inputs[1], y, x, p, b);
    ReLU(x, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < layer_output; j++)
            cat[i * cat_input + layer_inputs[0] + layer_inputs[1] + j] = x[i * layer_output + j];
    prep_params(params + param_offsets[2], layer_inputs[2], layer_outputs[2], p, b);
    matrix_multiplication_scalar(cat, p, y, b, N, layer_inputs[2]);
    ReLU(y, N);
    prep_params(params + param_offsets[3], layer_inputs[3], layer_outputs[3], p, b);
    matrix_multiplication_scalar(y, p, x, b, N, layer_inputs[3]);
    Sigmoid(x, N);
    float *res = (float *)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) res[i] = x[i * layer_output];
    free(params); free(x); free(y); free(cat); free(p); free(b);
    return res;
}
]])
endif()

# Install GNN model files as package data
file(GLOB LR_MODEL_FILES ${LR_DIR}/models/*.lr_gcn)
install(FILES ${LR_MODEL_FILES} DESTINATION chszlablib/models)

# =============================================================================
# KaMIS  (Maximum Independent Set — three isolated static libraries)
# =============================================================================
# KaMIS has three sub-projects (root, wmis, mmwis) each with their own copy of
# KaHIP and conflicting symbols.  We build each as a separate static lib.

set(KAMIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/KaMIS)

# ---------------------------------------------------------------------------
# kamis_redumis_static — unweighted MIS (ReduMIS + OnlineMIS)
# ---------------------------------------------------------------------------
set(KAMIS_ROOT_KAHIP_SOURCES
    ${KAMIS_DIR}/extern/KaHIP/lib/data_structure/graph_hierarchy.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/algorithms/strongly_connected_components.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/algorithms/topological_sort.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/algorithms/push_relabel.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/io/graph_io.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/tools/quality_metrics.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/tools/random_functions.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/tools/graph_extractor.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/tools/misc.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/tools/partition_snapshooter.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/graph_partitioner.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/coarsening.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/contraction.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/clustering/node_ordering.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/matching.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/random_matching.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/path.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/bipartition.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/uncoarsening.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/algorithms/cycle_search.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    ${KAMIS_DIR}/extern/KaHIP/interface/kaHIP_interface.cpp
)

set(KAMIS_ROOT_LIB_SOURCES
    ${KAMIS_DIR}/lib/tools/mis_log.cpp
    ${KAMIS_DIR}/lib/mis/ils/ils.cpp
    ${KAMIS_DIR}/lib/mis/ils/local_search.cpp
    ${KAMIS_DIR}/lib/mis/initial_mis/greedy_mis.cpp
    ${KAMIS_DIR}/lib/mis/initial_mis/greedy_vertex.cpp
    ${KAMIS_DIR}/lib/mis/initial_mis/random_mis.cpp
    ${KAMIS_DIR}/lib/mis/initial_mis/initial_mis.cpp
    ${KAMIS_DIR}/lib/data_structure/mis_permutation.cpp
    ${KAMIS_DIR}/lib/data_structure/candidate_list.cpp
    ${KAMIS_DIR}/lib/data_structure/operation_log.cpp
    ${KAMIS_DIR}/lib/data_structure/priority_queues/bucket_array.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/population_mis.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/reduction_evolution.cpp
    ${KAMIS_DIR}/lib/mis/hopcroft/bipartite_vertex_cover.cpp
    ${KAMIS_DIR}/lib/mis/kernel/branch_and_reduce_algorithm.cpp
    ${KAMIS_DIR}/lib/mis/kernel/modified.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/separator_pool.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/combine/combine.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/combine/cover_combine.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/combine/separator_combine.cpp
    ${KAMIS_DIR}/lib/mis/evolutionary/combine/multiway_combine.cpp
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/fast_reductions/src/full_reductions.cpp
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/fast_reductions/src/parallel_reductions.cpp
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/fast_reductions/src/MaximumMatching.cpp
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/LinearTime/MIS_sigmod_pub/Graph.cpp
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/LinearTime/MIS_sigmod_pub/Utility.cpp
    ${KAMIS_DIR}/extern/argtable3-3.0.3/argtable3.c
)

set(KAMIS_ONLINE_SOURCES
    ${KAMIS_DIR}/lib/mis/ils/online_ils.cpp
    ${KAMIS_DIR}/lib/mis/ils/local_search_online.cpp
    ${KAMIS_DIR}/lib/mis/initial_mis/greedy_mis_online.cpp
    ${KAMIS_DIR}/lib/data_structure/mis_permutation_online.cpp
)

add_library(kamis_redumis_static STATIC
    ${KAMIS_ROOT_KAHIP_SOURCES}
    ${KAMIS_ROOT_LIB_SOURCES}
    ${KAMIS_ONLINE_SOURCES}
)

target_include_directories(kamis_redumis_static PUBLIC
    ${KAMIS_DIR}/extern/argtable3-3.0.3
    ${KAMIS_DIR}
    ${KAMIS_DIR}/app
    ${KAMIS_DIR}/extern/KaHIP
    ${KAMIS_DIR}/extern/KaHIP/interface
    ${KAMIS_DIR}/extern/KaHIP/lib
    ${KAMIS_DIR}/extern/KaHIP/lib/data_structure
    ${KAMIS_DIR}/extern/KaHIP/lib/io
    ${KAMIS_DIR}/extern/KaHIP/lib/partition
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${KAMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements
    ${KAMIS_DIR}/extern/KaHIP/lib/tools
    ${KAMIS_DIR}/lib
    ${KAMIS_DIR}/lib/mis
    ${KAMIS_DIR}/lib/mis/evolutionary
    ${KAMIS_DIR}/lib/mis/evolutionary/combine
    ${KAMIS_DIR}/lib/mis/initial_mis
    ${KAMIS_DIR}/lib/data_structure
    ${KAMIS_DIR}/lib/io
    ${KAMIS_DIR}/lib/tools
    ${KAMIS_DIR}/lib/algorithms
)

target_compile_definitions(kamis_redumis_static PRIVATE OMIS)
target_compile_options(kamis_redumis_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kamis_redumis_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# kamis_wmis_static — weighted MIS exact (branch & reduce)
# ---------------------------------------------------------------------------
set(KAMIS_WMIS_DIR ${KAMIS_DIR}/wmis)

set(KAMIS_WMIS_KAHIP_SOURCES
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/data_structure/graph_hierarchy.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/algorithms/strongly_connected_components.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/algorithms/topological_sort.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/algorithms/push_relabel.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/io/graph_io.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/tools/quality_metrics.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/tools/random_functions.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/tools/graph_extractor.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/tools/misc.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/tools/partition_snapshooter.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/graph_partitioner.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/coarsening.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/contraction.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/matching.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/random_matching.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/path.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/clustering/node_ordering.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/bipartition.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/initial_partitioning/initial_node_separator.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/uncoarsening.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/area_bfs.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/node_separators/greedy_ns_local_search.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/node_separators/fm_ns_local_search.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/node_separators/localized_fm_ns_local_search.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/algorithms/cycle_search.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
)

set(KAMIS_WMIS_LIB_SOURCES
    ${KAMIS_WMIS_DIR}/lib/tools/mis_log.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/ils/ils.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/ils/local_search.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/initial_mis/greedy_mis.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/initial_mis/greedy_vertex.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/initial_mis/random_mis.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/initial_mis/initial_mis.cpp
    ${KAMIS_WMIS_DIR}/lib/data_structure/mis_permutation.cpp
    ${KAMIS_WMIS_DIR}/lib/data_structure/candidate_list.cpp
    ${KAMIS_WMIS_DIR}/lib/data_structure/operation_log.cpp
    ${KAMIS_WMIS_DIR}/lib/data_structure/priority_queues/bucket_array.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/kernel/branch_and_reduce_algorithm.cpp
    ${KAMIS_WMIS_DIR}/lib/mis/kernel/reductions.cpp
    ${KAMIS_DIR}/extern/argtable3-3.0.3/argtable3.c
)

add_library(kamis_wmis_static STATIC
    ${KAMIS_WMIS_KAHIP_SOURCES}
    ${KAMIS_WMIS_LIB_SOURCES}
)

target_include_directories(kamis_wmis_static PUBLIC
    ${KAMIS_WMIS_DIR}
    ${KAMIS_WMIS_DIR}/app
    ${KAMIS_WMIS_DIR}/extern/argtable3-3.0.3
    ${KAMIS_WMIS_DIR}/lib
    ${KAMIS_WMIS_DIR}/lib/data_structure
    ${KAMIS_WMIS_DIR}/lib/data_structure/priority_queues
    ${KAMIS_WMIS_DIR}/lib/mis
    ${KAMIS_WMIS_DIR}/lib/mis/ils
    ${KAMIS_WMIS_DIR}/lib/mis/initial_mis
    ${KAMIS_WMIS_DIR}/lib/mis/kernel
    ${KAMIS_WMIS_DIR}/lib/tools
    ${KAMIS_WMIS_DIR}/extern/KaHIP
    ${KAMIS_WMIS_DIR}/extern/KaHIP/app
    ${KAMIS_WMIS_DIR}/extern/KaHIP/interface
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/io
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/tools
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/algorithms
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/data_structure
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/data_structure/matrix
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/data_structure/priority_queues
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition
    ${KAMIS_WMIS_DIR}/extern/KaHIP/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
)

target_compile_options(kamis_wmis_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kamis_wmis_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Clang compatibility patches for KaMIS MMWIS struction sources.
# GCC allows explicit template instantiations after `using namespace X;`
# but Clang requires them inside `namespace X { ... }`.
# Also fix int*/unsigned int* mismatch in build_from_metis call.
# Patches are idempotent (string(REPLACE) is a no-op if pattern not found).
# ---------------------------------------------------------------------------
set(_STRUCTION_DIR ${KAMIS_DIR}/mmwis/extern/struction/lib/mis/kernel)
set(_MMWIS_KERNEL_DIR ${KAMIS_DIR}/mmwis/lib/mis/kernel)

# --- struction_reductions.cpp: wrap each instantiation line individually ---
# Guard: only patch if not already patched (avoid nested namespace wrapping)
file(READ ${_STRUCTION_DIR}/struction_reductions.cpp _SRC)
string(FIND "${_SRC}" "namespace struction { template class" _ALREADY_PATCHED)
if(_ALREADY_PATCHED EQUAL -1)
    string(REPLACE "template class iterative_struction<" "namespace struction { template class iterative_struction<" _SRC "${_SRC}")
    string(REPLACE "template class blow_up_struction<"    "namespace struction { template class blow_up_struction<"    _SRC "${_SRC}")
    string(REGEX REPLACE "(namespace struction \\{ template class [^;]+;)" "\\1 }" _SRC "${_SRC}")
    file(WRITE ${_STRUCTION_DIR}/struction_reductions.cpp "${_SRC}")
endif()

# --- extended_struction.cpp ---
file(READ ${_STRUCTION_DIR}/extended_struction.cpp _SRC)
string(FIND "${_SRC}" "namespace struction { template class" _ALREADY_PATCHED)
if(_ALREADY_PATCHED EQUAL -1)
    string(REPLACE "template class extended_struction<" "namespace struction { template class extended_struction<" _SRC "${_SRC}")
    string(REGEX REPLACE "(namespace struction \\{ template class [^;]+;)" "\\1 }" _SRC "${_SRC}")
    file(WRITE ${_STRUCTION_DIR}/extended_struction.cpp "${_SRC}")
endif()

# --- original_struction.cpp ---
file(READ ${_STRUCTION_DIR}/original_struction.cpp _SRC)
string(FIND "${_SRC}" "namespace struction { template class" _ALREADY_PATCHED)
if(_ALREADY_PATCHED EQUAL -1)
    string(REPLACE "template class original_struction<" "namespace struction { template class original_struction<" _SRC "${_SRC}")
    string(REGEX REPLACE "(namespace struction \\{ template class [^;]+;)" "\\1 }" _SRC "${_SRC}")
    file(WRITE ${_STRUCTION_DIR}/original_struction.cpp "${_SRC}")
endif()

# --- reduction_evolution.cpp: namespace + default argument redefinition ---
set(_MMWIS_EVO_DIR ${KAMIS_DIR}/mmwis/lib/mis/evolutionary)
file(READ ${_MMWIS_EVO_DIR}/reduction_evolution.cpp _SRC)
# Clang rejects default arg redefinition in definition when already in declaration
string(REPLACE "bool recursive = false," "bool recursive," _SRC "${_SRC}")
# Wrap explicit instantiation in namespace mmwis (idempotent: skip if already wrapped)
string(FIND "${_SRC}" "namespace mmwis { template class reduction_evolution" _ALREADY_PATCHED)
if(_ALREADY_PATCHED EQUAL -1)
    string(REPLACE
        "template class reduction_evolution<branch_and_reduce_algorithm>;"
        "namespace mmwis { template class reduction_evolution<branch_and_reduce_algorithm>; }"
        _SRC "${_SRC}")
endif()
file(WRITE ${_MMWIS_EVO_DIR}/reduction_evolution.cpp "${_SRC}")

# --- branch_and_reduce_algorithm.cpp: int*/unsigned int* mismatch ---
file(READ ${_MMWIS_KERNEL_DIR}/branch_and_reduce_algorithm.cpp _SRC)
string(REPLACE
    "G.build_from_metis(status.remaining_nodes, &xadj[0], &adjncy[0])"
    "G.build_from_metis(status.remaining_nodes, reinterpret_cast<int*>(&xadj[0]), reinterpret_cast<int*>(&adjncy[0]))"
    _SRC "${_SRC}")
file(WRITE ${_MMWIS_KERNEL_DIR}/branch_and_reduce_algorithm.cpp "${_SRC}")

# --- MaximumMatching.cpp: missing omp.h include (clang needs explicit include) ---
set(_PARFASTKER_SRC_DIR ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/fast_reductions/src)
file(READ ${_PARFASTKER_SRC_DIR}/MaximumMatching.cpp _SRC)
string(REPLACE "#include <parallel/numeric>" "#include <omp.h>\n#include <parallel/numeric>" _SRC "${_SRC}")
file(WRITE ${_PARFASTKER_SRC_DIR}/MaximumMatching.cpp "${_SRC}")

# ---------------------------------------------------------------------------
# kamis_mmwis_static — weighted MIS memetic evolutionary (MMWIS)
# ---------------------------------------------------------------------------
set(KAMIS_MMWIS_DIR ${KAMIS_DIR}/mmwis)
set(KAMIS_MMWIS_KAHIP ${KAMIS_MMWIS_DIR}/extern/KaHIP)
set(KAMIS_MMWIS_MODKAHIP ${KAMIS_MMWIS_DIR}/lib/modKaHIPFiles)

# KaHIP sources for mmwis (from modKaHIPFiles CMakeLists — uses mmwis/extern/KaHIP
# for most files but overrides graph_access.h and kaHIP_interface from modKaHIPFiles)
set(KAMIS_MMWIS_KAHIP_SOURCES
    ${KAMIS_MMWIS_KAHIP}/lib/io/graph_io.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/tools/graph_extractor.cpp
    ${KAMIS_MMWIS_MODKAHIP}/interface/kaHIP_interface.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/algorithms/strongly_connected_components.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/data_structure/graph_hierarchy.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/algorithms/topological_sort.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/algorithms/push_relabel.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/tools/quality_metrics.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/tools/misc.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/tools/partition_snapshooter.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/tools/random_functions.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/graph_partitioner.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/coarsening.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/contraction.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/matching/matching.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/matching/random_matching.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/matching/gpa/path.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/clustering/node_ordering.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/initial_partitioning/bipartition.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/initial_partitioning/initial_node_separator.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/uncoarsening.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/separator/area_bfs.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/node_separators/greedy_ns_local_search.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/node_separators/fm_ns_local_search.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/node_separators/localized_fm_ns_local_search.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/algorithms/cycle_search.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    ${KAMIS_MMWIS_KAHIP}/extern/argtable3-3.2.2/argtable3.c
    # mapping + node_ordering (needed by kaHIP_interface)
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/local_search_mapping.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/full_search_space.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/full_search_space_pruned.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/communication_graph_search_space.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/fast_construct_mapping.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/construct_distance_matrix.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/mapping_algorithms.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/mapping/construct_mapping.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/node_ordering/min_degree_ordering.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/node_ordering/nested_dissection.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/node_ordering/ordering_tools.cpp
    ${KAMIS_MMWIS_KAHIP}/lib/node_ordering/reductions.cpp
)

set(KAMIS_MMWIS_LIB_SOURCES
    ${KAMIS_MMWIS_DIR}/lib/tools/mmwis_log.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/ils/ils.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/ils/local_search.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/hils/Solution.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/greedy_mis.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/greedy_vertex.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/greedy_mwis.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/greedy_weighted_vertex.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/random_mis.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/initial_mis.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis/cyclicFast.cpp
    ${KAMIS_MMWIS_DIR}/lib/data_structure/mis_permutation.cpp
    ${KAMIS_MMWIS_DIR}/lib/data_structure/candidate_list.cpp
    ${KAMIS_MMWIS_DIR}/lib/data_structure/operation_log.cpp
    ${KAMIS_MMWIS_DIR}/lib/data_structure/priority_queues/bucket_array.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/kernel/branch_and_reduce_algorithm.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/kernel/cout_handler.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/kernel/reductions.cpp
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel/struction_reductions.cpp
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel/struction_branch_and_reduce_algorithm.cpp
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel/extended_struction.cpp
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel/key_functions.cpp
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel/mwis_finder.cpp
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel/original_struction.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/max_flow/bipartite_weighted_vertex_cover.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/population_mis.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/reduction_evolution.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/separator_pool.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/combine/combine.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/combine/cover_combine.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/combine/separator_combine.cpp
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/combine/multiway_combine.cpp
    ${KAMIS_MMWIS_DIR}/extern/argtable3-3.2.1/argtable3.c
)

add_library(kamis_mmwis_static STATIC
    ${KAMIS_MMWIS_KAHIP_SOURCES}
    ${KAMIS_MMWIS_LIB_SOURCES}
)

# modKaHIPFiles headers take priority (overrides graph_access.h, kaHIP_interface.h)
target_include_directories(kamis_mmwis_static PUBLIC
    ${KAMIS_MMWIS_MODKAHIP}/interface
    ${KAMIS_MMWIS_MODKAHIP}/lib
    ${KAMIS_MMWIS_MODKAHIP}/lib/data_structure
    ${KAMIS_MMWIS_DIR}/lib/data_structure
    ${KAMIS_MMWIS_DIR}/lib/data_structure/priority_queues
    ${KAMIS_MMWIS_KAHIP}/app
    ${KAMIS_MMWIS_KAHIP}/lib
    ${KAMIS_MMWIS_KAHIP}/lib/io
    ${KAMIS_MMWIS_KAHIP}/lib/tools
    ${KAMIS_MMWIS_KAHIP}/lib/algorithms
    ${KAMIS_MMWIS_KAHIP}/lib/partition
    ${KAMIS_MMWIS_KAHIP}/lib/data_structure/priority_queues
    ${KAMIS_MMWIS_KAHIP}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${KAMIS_MMWIS_DIR}/app
    ${KAMIS_MMWIS_DIR}/lib
    ${KAMIS_MMWIS_DIR}/lib/mis
    ${KAMIS_MMWIS_DIR}/lib/mis/ils
    ${KAMIS_MMWIS_DIR}/lib/mis/hils
    ${KAMIS_MMWIS_DIR}/lib/mis/initial_mis
    ${KAMIS_MMWIS_DIR}/lib/mis/kernel
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary
    ${KAMIS_MMWIS_DIR}/lib/mis/evolutionary/combine
    ${KAMIS_MMWIS_DIR}/lib/mis/max_flow
    ${KAMIS_MMWIS_DIR}/lib/tools
    ${KAMIS_MMWIS_DIR}/extern
    ${KAMIS_MMWIS_DIR}/extern/argtable3-3.2.1
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/mis/kernel
    ${KAMIS_MMWIS_DIR}/extern/struction/lib/data_structure
)

target_compile_options(kamis_mmwis_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kamis_mmwis_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# ScalableCorrelationClustering (SCC)
# =============================================================================
set(SCC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/SCC)

add_library(scc_static STATIC
    ${SCC_DIR}/extern/argtable3-3.0.3/argtable3.c
    ${SCC_DIR}/lib/clustering/signed_graph_clusterer.cpp
    ${SCC_DIR}/lib/clustering/coarsening/clustering/node_ordering.cpp
    ${SCC_DIR}/lib/clustering/coarsening/clustering/size_constraint_label_propagation.cpp
    ${SCC_DIR}/lib/clustering/coarsening/coarsening.cpp
    ${SCC_DIR}/lib/clustering/coarsening/contraction.cpp
    ${SCC_DIR}/lib/tools/graph_extractor.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/refinement.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/uncoarsening.cpp
    ${SCC_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${SCC_DIR}/lib/io/graph_io.cpp
    ${SCC_DIR}/lib/tools/quality_metrics.cpp
    ${SCC_DIR}/lib/tools/random_functions.cpp
    ${SCC_DIR}/tools/tools.cpp
)

target_include_directories(scc_static PUBLIC
    ${SCC_DIR}
    ${SCC_DIR}/app
    ${SCC_DIR}/extern/argtable3-3.0.3
    ${SCC_DIR}/extern/andres_graph/include
    ${SCC_DIR}/lib
    ${SCC_DIR}/lib/io
    ${SCC_DIR}/lib/partition
    ${SCC_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${SCC_DIR}/lib/tools
)

target_compile_options(scc_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(scc_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# SCC evolutionary (memetic) — reuses libclustering + adds evolutionary sources
# Uses a fake mpi.h shim (single-process stub) instead of real MPI.
add_library(scc_evo_static STATIC
    # Base clustering sources (same as scc_static)
    ${SCC_DIR}/extern/argtable3-3.0.3/argtable3.c
    ${SCC_DIR}/lib/clustering/signed_graph_clusterer.cpp
    ${SCC_DIR}/lib/clustering/coarsening/clustering/node_ordering.cpp
    ${SCC_DIR}/lib/clustering/coarsening/clustering/size_constraint_label_propagation.cpp
    ${SCC_DIR}/lib/clustering/coarsening/coarsening.cpp
    ${SCC_DIR}/lib/clustering/coarsening/contraction.cpp
    ${SCC_DIR}/lib/tools/graph_extractor.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/refinement/refinement.cpp
    ${SCC_DIR}/lib/clustering/uncoarsening/uncoarsening.cpp
    ${SCC_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${SCC_DIR}/lib/io/graph_io.cpp
    ${SCC_DIR}/lib/tools/quality_metrics.cpp
    ${SCC_DIR}/lib/tools/random_functions.cpp
    ${SCC_DIR}/tools/tools.cpp
    # Evolutionary sources
    ${SCC_DIR}/lib/algorithms/cycle_search.cpp
    ${SCC_DIR}/lib/algorithms/strongly_connected_components.cpp
    ${SCC_DIR}/lib/clustering_evolutionary/evolutionary_signed_graph_clusterer.cpp
    ${SCC_DIR}/lib/clustering_evolutionary/population.cpp
    ${SCC_DIR}/lib/clustering_evolutionary/exchange/exchanger.cpp
    ${SCC_DIR}/lib/tools/graph_communication.cpp
    ${SCC_DIR}/lib/tools/mpi_tools.cpp
)

# Fake mpi.h MUST come first so #include <mpi.h> finds our stub
target_include_directories(scc_evo_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/scc_mpi
    ${SCC_DIR}
    ${SCC_DIR}/app
    ${SCC_DIR}/extern/argtable3-3.0.3
    ${SCC_DIR}/extern/andres_graph/include
    ${SCC_DIR}/lib
    ${SCC_DIR}/lib/io
    ${SCC_DIR}/lib/partition
    ${SCC_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${SCC_DIR}/lib/tools
)

target_compile_options(scc_evo_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(scc_evo_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# HeiOrient  (header-only — edge orientation algorithms)
# =============================================================================
set(HEIORIENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiOrient)

add_library(heiorient INTERFACE)
target_include_directories(heiorient INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/heiOrient_absl   # abseil shim (must come first)
    ${HEIORIENT_DIR}                                     # HeiOrient root (Bazel-style includes)
)
target_compile_features(heiorient INTERFACE cxx_std_17)

# --- Patch: MSVC rejects narrowing conversions inside brace init ---
file(READ ${HEIORIENT_DIR}/edge_orientation/ds/smaller_graph.h _SRC)
string(REPLACE "IndexType start{_nodes_offset.size() - 1}"
    "IndexType start(static_cast<IT>(_nodes_offset.size() - 1))" _SRC "${_SRC}")
string(REPLACE "InsideArrayRefType{edge - flat_edge_info.data()}"
    "InsideArrayRefType(static_cast<IARF>(edge - flat_edge_info.data()))" _SRC "${_SRC}" )
string(REPLACE "IndexType{out_s.size() - 1}"
    "IndexType(static_cast<IT>(out_s.size() - 1))" _SRC "${_SRC}")
file(WRITE ${HEIORIENT_DIR}/edge_orientation/ds/smaller_graph.h "${_SRC}")

# =============================================================================
# KaHIP Evolutionary (kaffpaE) — same KaHIP/lib as CLI, pseudo-MPI shim
# =============================================================================
set(KAHIPE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/KaHIP)

add_library(kahipe_static STATIC
    ${KAHIPE_DIR}/interface/kaHIP_interface.cpp
    # Core KaHIP sources (same lib/ as CLI binary)
    ${KAHIPE_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${KAHIPE_DIR}/lib/algorithms/strongly_connected_components.cpp
    ${KAHIPE_DIR}/lib/algorithms/topological_sort.cpp
    ${KAHIPE_DIR}/lib/algorithms/push_relabel.cpp
    ${KAHIPE_DIR}/lib/algorithms/cycle_search.cpp
    ${KAHIPE_DIR}/lib/io/graph_io.cpp
    ${KAHIPE_DIR}/lib/tools/quality_metrics.cpp
    ${KAHIPE_DIR}/lib/tools/random_functions.cpp
    ${KAHIPE_DIR}/lib/tools/graph_extractor.cpp
    ${KAHIPE_DIR}/lib/tools/misc.cpp
    ${KAHIPE_DIR}/lib/tools/partition_snapshooter.cpp
    ${KAHIPE_DIR}/lib/partition/graph_partitioner.cpp
    ${KAHIPE_DIR}/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/coarsening.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/contraction.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/matching/matching.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/matching/random_matching.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/matching/gpa/path.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/clustering/node_ordering.cpp
    ${KAHIPE_DIR}/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${KAHIPE_DIR}/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${KAHIPE_DIR}/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${KAHIPE_DIR}/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${KAHIPE_DIR}/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/initial_partitioning/bipartition.cpp
    ${KAHIPE_DIR}/lib/partition/initial_partitioning/initial_node_separator.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/uncoarsening.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/separator/area_bfs.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/node_separators/greedy_ns_local_search.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/node_separators/fm_ns_local_search.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/node_separators/localized_fm_ns_local_search.cpp
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    # Parallel MH sources (with MPI shim)
    ${KAHIPE_DIR}/lib/parallel_mh/galinier_combine/gal_combine.cpp
    ${KAHIPE_DIR}/lib/parallel_mh/galinier_combine/construct_partition.cpp
    ${KAHIPE_DIR}/lib/parallel_mh/parallel_mh_async.cpp
    ${KAHIPE_DIR}/lib/parallel_mh/population.cpp
    ${KAHIPE_DIR}/lib/parallel_mh/exchange/exchanger.cpp
    ${KAHIPE_DIR}/lib/tools/graph_communication.cpp
    ${KAHIPE_DIR}/lib/tools/mpi_tools.cpp
    ${KAHIPE_DIR}/extern/argtable3-3.2.2/argtable3.c
)

# Fake mpi.h MUST come first so #include <mpi.h> finds our stub
target_include_directories(kahipe_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/scc_mpi
    ${KAHIPE_DIR}/interface
    ${KAHIPE_DIR}/lib
    ${KAHIPE_DIR}/lib/io
    ${KAHIPE_DIR}/lib/partition
    ${KAHIPE_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${KAHIPE_DIR}/lib/tools
    ${KAHIPE_DIR}
    ${KAHIPE_DIR}/app
    ${KAHIPE_DIR}/extern/argtable3-3.2.2
)

target_compile_definitions(kahipe_static PRIVATE -DMODE_KAFFPAE)
target_compile_options(kahipe_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kahipe_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# fpt-max-cut  (FPT kernelization + MQLib solver for Max-Cut)
# =============================================================================
# MQLib was built above (before the -include cstdint workaround).
# Here we build the fpt-max-cut core as a static library.

# --- fpt-max-cut static library ---
add_library(fpt_maxcut_static STATIC
    ${FPT_MAXCUT_DIR}/src/mc-graph.cpp
    ${FPT_MAXCUT_DIR}/src/utils.cpp
    ${FPT_MAXCUT_DIR}/src/output-filter.cpp
)
target_include_directories(fpt_maxcut_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/fpt_maxcut   # filesystem shim (must come first)
    ${FPT_MAXCUT_DIR}                                # project root (for ./src/output-filter.hpp)
    ${FPT_MAXCUT_DIR}/src                            # source headers
    ${MQLIB_DIR}/include                             # MQLib headers
)
target_link_libraries(fpt_maxcut_static PUBLIC mqlib_static)
target_compile_options(fpt_maxcut_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)
target_compile_definitions(fpt_maxcut_static PRIVATE NDEBUG)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fpt_maxcut_static PUBLIC OpenMP::OpenMP_CXX)
endif()
find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(fpt_maxcut_static PUBLIC Threads::Threads)
endif()

# =============================================================================
# HeidelbergMotifClustering  (SOCIAL + LMCHGP local motif clustering)
# =============================================================================
set(MOTIF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeidelbergMotifClustering/SOCIAL)

add_library(motif_static STATIC
    ${MOTIF_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${MOTIF_DIR}/lib/algorithms/strongly_connected_components.cpp
    ${MOTIF_DIR}/lib/algorithms/topological_sort.cpp
    ${MOTIF_DIR}/lib/algorithms/bfs_depth.cpp
    ${MOTIF_DIR}/lib/algorithms/triangle_listing.cpp
    ${MOTIF_DIR}/lib/algorithms/mqi.cpp
    ${MOTIF_DIR}/lib/algorithms/push_relabel.cpp
    ${MOTIF_DIR}/lib/algorithms/cycle_search.cpp
    ${MOTIF_DIR}/lib/io/graph_io.cpp
    ${MOTIF_DIR}/lib/tools/quality_metrics.cpp
    ${MOTIF_DIR}/lib/tools/random_functions.cpp
    ${MOTIF_DIR}/lib/tools/graph_extractor.cpp
    ${MOTIF_DIR}/lib/tools/misc.cpp
    ${MOTIF_DIR}/lib/tools/partition_snapshooter.cpp
    ${MOTIF_DIR}/lib/partition/graph_partitioner.cpp
    ${MOTIF_DIR}/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/coarsening.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/contraction.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/matching/matching.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/matching/random_matching.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/matching/gpa/path.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/clustering/node_ordering.cpp
    ${MOTIF_DIR}/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${MOTIF_DIR}/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${MOTIF_DIR}/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${MOTIF_DIR}/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${MOTIF_DIR}/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${MOTIF_DIR}/lib/partition/initial_partitioning/bipartition.cpp
    ${MOTIF_DIR}/lib/partition/initial_partitioning/initial_node_separator.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/uncoarsening.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/separator/area_bfs.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_motif_cond_refinement/kway_motif_cond_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_motif_cond_refinement/kway_motif_cond_refinement_core.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_motif_cond_refinement/kway_motif_cond_refinement_commons.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/kway_motif_cond_refinement/multitry_kway_fm_motif_cond.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/node_separators/greedy_ns_local_search.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/node_separators/fm_ns_local_search.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/node_separators/localized_fm_ns_local_search.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    ${MOTIF_DIR}/lib/mapping/local_search_mapping.cpp
    ${MOTIF_DIR}/lib/mapping/full_search_space.cpp
    ${MOTIF_DIR}/lib/mapping/full_search_space_pruned.cpp
    ${MOTIF_DIR}/lib/mapping/communication_graph_search_space.cpp
    ${MOTIF_DIR}/lib/mapping/fast_construct_mapping.cpp
    ${MOTIF_DIR}/lib/mapping/construct_distance_matrix.cpp
    ${MOTIF_DIR}/lib/mapping/mapping_algorithms.cpp
    ${MOTIF_DIR}/lib/mapping/construct_mapping.cpp
    ${MOTIF_DIR}/extern/argtable3-3.0.3/argtable3.c
)

target_include_directories(motif_static PUBLIC
    ${MOTIF_DIR}
    ${MOTIF_DIR}/app
    ${MOTIF_DIR}/extern/argtable3-3.0.3
    ${MOTIF_DIR}/lib
    ${MOTIF_DIR}/lib/io
    ${MOTIF_DIR}/lib/partition
    ${MOTIF_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${MOTIF_DIR}/lib/tools
)

target_compile_options(motif_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)
target_compile_definitions(motif_static PRIVATE
    -DNDEBUG
    -DMODE64BITEDGES
    -DPOINTER64=1
    -DMODE_MOTIF_CLUSTERING_GRAPH
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(motif_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# HeiStream  (streaming graph partitioner — full BuffCut + Fennel)
# =============================================================================
set(HEISTREAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiStream)

# FetchContent for readerwriterqueue (lock-free queue used by parallel mode)
include(FetchContent)
FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue
  GIT_TAG        master
)
FetchContent_MakeAvailable(readerwriterqueue)

# --- Patches: Windows / MSVC compatibility for HeiStream ---
# sparsehash: __attribute__((unused)) not supported on MSVC
file(READ ${HEISTREAM_DIR}/extern/sparsehash/src/sparsehash/internal/hashtable-common.h _SRC)
string(REPLACE
    "__attribute__((unused)) typedef SparsehashCompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]"
    "typedef SparsehashCompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]"
    _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/extern/sparsehash/src/sparsehash/internal/hashtable-common.h "${_SRC}")

# timer.h: sys/resource.h, sys/time.h, unistd.h are Unix-only
file(READ ${HEISTREAM_DIR}/lib/tools/timer.h _SRC)
string(REPLACE
    "#include <sys/resource.h>\n#include <sys/time.h>\n#include <unistd.h>"
    "#ifdef _WIN32\n#include <windows.h>\n#else\n#include <sys/resource.h>\n#include <sys/time.h>\n#include <unistd.h>\n#endif"
    _SRC "${_SRC}")
string(REPLACE
    "        struct timeval tp;\n        gettimeofday(&tp, nullptr);\n        return double(tp.tv_sec) + tp.tv_usec / 1000000.;"
    "#ifdef _WIN32\n        LARGE_INTEGER freq, cnt;\n        QueryPerformanceFrequency(&freq);\n        QueryPerformanceCounter(&cnt);\n        return double(cnt.QuadPart) / double(freq.QuadPart);\n#else\n        struct timeval tp;\n        gettimeofday(&tp, nullptr);\n        return double(tp.tv_sec) + tp.tv_usec / 1000000.;\n#endif"
    _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/tools/timer.h "${_SRC}")

# complete_boundary.h: execinfo.h is Unix-only (unused include)
file(READ ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.h _SRC)
string(REPLACE "#include <execinfo.h>" "#ifndef _WIN32\n#include <execinfo.h>\n#endif" _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.h "${_SRC}")

# maxNodeHeap.h: execinfo.h is Unix-only (unused include)
file(READ ${HEISTREAM_DIR}/lib/data_structure/priority_queues/maxNodeHeap.h _SRC)
string(REPLACE "#include <execinfo.h>" "#ifndef _WIN32\n#include <execinfo.h>\n#endif" _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/data_structure/priority_queues/maxNodeHeap.h "${_SRC}")

# compare_rating.h: std::binary_function removed in C++17
file(READ ${HEISTREAM_DIR}/lib/partition/coarsening/matching/gpa/compare_rating.h _SRC)
string(REPLACE
    "class compare_rating : public std::binary_function<EdgeRatingType, EdgeRatingType, bool> {"
    "class compare_rating {"
    _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/partition/coarsening/matching/gpa/compare_rating.h "${_SRC}")

# --- Patch: replace exit(0) with throw in HeiStream core files ---
# When used as a library (via pybind11), exit(0) kills the Python process.
# Replace with exceptions so errors propagate as Python RuntimeError.

# model_graph_ops.cpp — exit() kills the Python process
file(READ ${HEISTREAM_DIR}/lib/partition/model/model_graph_ops.cpp _SRC)
string(FIND "${_SRC}" "#include <stdexcept>" _HAS_STDEXCEPT)
if(_HAS_STDEXCEPT EQUAL -1)
    string(REPLACE "#include <iostream>" "#include <iostream>\n#include <stdexcept>" _SRC "${_SRC}")
endif()
string(REPLACE "exit(0)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
string(REPLACE "exit(1)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/partition/model/model_graph_ops.cpp "${_SRC}")

# node_batch_model_builder.cpp — exit() kills the Python process
file(READ ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/model/node_batch_model_builder.cpp _SRC)
string(FIND "${_SRC}" "#include <stdexcept>" _HAS_STDEXCEPT)
if(_HAS_STDEXCEPT EQUAL -1)
    string(REPLACE "#include <iostream>" "#include <iostream>\n#include <stdexcept>" _SRC "${_SRC}")
endif()
string(REPLACE "exit(0)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
string(REPLACE "exit(1)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/model/node_batch_model_builder.cpp "${_SRC}")

# edge_batch_model_builder.cpp — exit() kills the Python process
file(READ ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/model/edge_batch_model_builder.cpp _SRC)
string(FIND "${_SRC}" "#include <stdexcept>" _HAS_STDEXCEPT)
if(_HAS_STDEXCEPT EQUAL -1)
    string(REPLACE "#include <iostream>" "#include <iostream>\n#include <stdexcept>" _SRC "${_SRC}")
endif()
string(REPLACE "exit(0)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
string(REPLACE "exit(1)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/model/edge_batch_model_builder.cpp "${_SRC}")

# graph_io.cpp — exit() kills the Python process
file(READ ${HEISTREAM_DIR}/lib/io/graph_io.cpp _SRC)
string(FIND "${_SRC}" "#include <stdexcept>" _HAS_STDEXCEPT)
if(_HAS_STDEXCEPT EQUAL -1)
    string(REPLACE "#include <iostream>" "#include <iostream>\n#include <stdexcept>" _SRC "${_SRC}")
endif()
string(REPLACE "exit(0)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
string(REPLACE "exit(1)" "throw std::runtime_error(\"HeiStream error (see stderr)\")" _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/lib/io/graph_io.cpp "${_SRC}")

# Build flatbuffers from HeiStream's bundled copy
add_subdirectory(${HEISTREAM_DIR}/extern/flatbuffers
    ${CMAKE_CURRENT_BINARY_DIR}/heistream-flatbuffers-build
    EXCLUDE_FROM_ALL)

set(HEISTREAM_SOURCES
    ${HEISTREAM_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${HEISTREAM_DIR}/lib/graph/algorithms/strongly_connected_components.cpp
    ${HEISTREAM_DIR}/lib/graph/algorithms/topological_sort.cpp
    ${HEISTREAM_DIR}/lib/graph/algorithms/push_relabel.cpp
    ${HEISTREAM_DIR}/lib/io/graph_io.cpp
    ${HEISTREAM_DIR}/lib/io/stream_reader.cpp
    ${HEISTREAM_DIR}/lib/io/output/flatbuffer_log_writer.cpp
    ${HEISTREAM_DIR}/lib/io/output/partition_output_writer.cpp
    ${HEISTREAM_DIR}/lib/io/edge_stream_reader.cpp
    ${HEISTREAM_DIR}/lib/io/node_stream_reader.cpp
    ${HEISTREAM_DIR}/lib/partition/model/model_graph_ops.cpp
    ${HEISTREAM_DIR}/lib/partition/model/model_partition_mapper.cpp
    ${HEISTREAM_DIR}/lib/partition/assignment/edge_stream_assign.cpp
    ${HEISTREAM_DIR}/lib/partition/assignment/node_stream_assign.cpp
    ${HEISTREAM_DIR}/lib/partition/metrics/edge_partitioning_quality_eval.cpp
    ${HEISTREAM_DIR}/lib/partition/metrics/node_partitioning_quality_eval.cpp
    ${HEISTREAM_DIR}/lib/partition/stages/stream_partition_stages.cpp
    ${HEISTREAM_DIR}/lib/partition/stages/batch_partition_execution.cpp
    ${HEISTREAM_DIR}/lib/partition/graph_partitioner.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/coarsening.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/contraction.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/matching/matching.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/matching/random_matching.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/matching/gpa/path.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/clustering/node_ordering.cpp
    ${HEISTREAM_DIR}/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${HEISTREAM_DIR}/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${HEISTREAM_DIR}/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${HEISTREAM_DIR}/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${HEISTREAM_DIR}/lib/partition/initial_partitioning/init_fennel.cpp
    ${HEISTREAM_DIR}/lib/partition/uncoarsening/uncoarsening.cpp
    ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/refinement.cpp
    ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${HEISTREAM_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/model/node_batch_model_builder.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/execution/node_execution_common.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/execution/node_execution_priority_buffer_parallel.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/execution/node_execution_priority_buffer_sequential.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/execution/node_execution_no_buffer.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/execution/node_execution_direct.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/setup/node_partitioning_setup.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/mode/node_partitioning_mode_resolver.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/mode/node_partitioning_mode_prepare.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/scheduler/node_partitioning_scheduler.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/evaluation/node_partitioning_evaluator.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/reporting/node_partitioning_reporter.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/node_partitioning_algorithm.cpp
    ${HEISTREAM_DIR}/lib/algorithms/node_partitioning/factory/node_partitioning_algorithm_factory.cpp
    ${HEISTREAM_DIR}/lib/algorithms/bootstrap/register_builtin_algorithms.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/model/edge_batch_model_builder.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/setup/edge_partitioning_setup.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/mode/edge_partitioning_mode_resolver.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/scheduler/edge_partitioning_scheduler.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/evaluation/edge_partitioning_evaluator.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/reporting/edge_partitioning_reporter.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/edge_partitioning_algorithm.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/factory/edge_partitioning_algorithm_factory.cpp
    ${HEISTREAM_DIR}/lib/algorithms/edge_partitioning/execution/edge_execution.cpp
    ${HEISTREAM_DIR}/lib/core/factory/algorithm_registry.cpp
    ${HEISTREAM_DIR}/lib/core/setup/stream_run_setup.cpp
    ${HEISTREAM_DIR}/lib/core/pipeline/algorithm_pipeline.cpp
    ${HEISTREAM_DIR}/lib/core/pipeline/stream_engine.cpp
    ${HEISTREAM_DIR}/lib/tools/quality_metrics.cpp
    ${HEISTREAM_DIR}/lib/tools/random_functions.cpp
    ${HEISTREAM_DIR}/lib/tools/graph_extractor.cpp
    ${HEISTREAM_DIR}/lib/tools/misc.cpp
    ${HEISTREAM_DIR}/lib/tools/partition_snapshooter.cpp
    ${HEISTREAM_DIR}/lib/tools/stream_util.cpp
    ${HEISTREAM_DIR}/extern/argtable3-3.0.3/argtable3.c
)

add_library(heistream_static STATIC ${HEISTREAM_SOURCES})
target_include_directories(heistream_static PUBLIC
    ${HEISTREAM_DIR}
    ${HEISTREAM_DIR}/extern/argtable3-3.0.3
    ${HEISTREAM_DIR}/extern/flatbuffers/include
    ${HEISTREAM_DIR}/extern/sparsehash/src
    ${HEISTREAM_DIR}/lib
    ${HEISTREAM_DIR}/lib/io
    ${HEISTREAM_DIR}/lib/partition
    ${HEISTREAM_DIR}/lib/tools
    ${readerwriterqueue_SOURCE_DIR}
)
target_compile_definitions(heistream_static PUBLIC
    -DMODE64BITEDGES
    -DPOINTER64=1
    -DMODE_STREAMPARTITION
)
target_compile_options(heistream_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc>)
target_link_libraries(heistream_static PUBLIC flatbuffers readerwriterqueue)

if(OpenMP_CXX_FOUND)
    target_link_libraries(heistream_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# CluStRE  (Streaming Graph Clustering — modularity / leiden)
# =============================================================================
set(CLUSTRE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/CluStRE)

# --- Patch CluStRE sources for portability (ARM, no abseil, no cpi) ---
# Write patch script to temp file to avoid cmake string escaping issues with regex.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_patch_clustre.py" [=[
import pathlib, re, sys

d = pathlib.Path(sys.argv[1])

# --- definitions.h: remove x86intrin.h, make crc32 always use fnv2a ---
f = d / 'lib/definitions.h'
if f.exists():
    t = f.read_text()
    t = t.replace('#include <x86intrin.h>', '// #include <x86intrin.h>  // removed for ARM')
    t = re.sub(
        r'#ifdef __SSE4_2__.*?_mm_crc32_u64.*?#else',
        '#if 0  // SSE4.2 removed for portability\n\treturn 0;\n#else',
        t, flags=re.DOTALL)
    f.write_text(t)

# --- partition_config.h: remove cpi + ExternalPQ includes ---
f = d / 'lib/partition/partition_config.h'
if f.exists():
    t = f.read_text()
    if 'patched out' not in t:
        t = t.replace('#include "cpi/run_length_compression.hpp"',
                       '// #include "cpi/run_length_compression.hpp"  // patched out')
        t = t.replace('#include "data_structure/ExternalPQ.h"',
                       '// #include "data_structure/ExternalPQ.h"  // patched out\nclass ExternalPQ;')
        f.write_text(t)

# --- vertex_partitioning.h/.cpp: replace absl, fix clang errors ---
f = d / 'lib/partition/onepass_partitioning/vertex_partitioning.h'
if f.exists():
    t = f.read_text()
    # Idempotent: replace absl include (clean up any prior repeated patching)
    t = re.sub(r'(?://\s*)*#include "absl/container/flat_hash_map.h"(?:\s*//\s*patched out)*',
               '// #include "absl/container/flat_hash_map.h"  // patched out', t)
    t = t.replace('absl::flat_hash_map', 'std::unordered_map')
    # Fix: bare 'return;' in non-void set_decision (clang error)
    t = re.sub(r'(set_decision\(PartitionID block,.*?my_thread, degree\);)\s*return;',
               r'\1 return block;', t, flags=re.DOTALL)
    # Fix: const correctness in get_block_address (declaration + definition)
    # Idempotent: collapse any multiple 'const' to single 'const'
    t = re.sub(r'(?:const\s+)+floating_block \*get_block_address',
               'const floating_block *get_block_address', t)
    t = re.sub(r'(?:const\s+)+floating_block \*block_pointer',
               'const floating_block *block_pointer', t)
    # Also fix the non-const original if still present
    t = re.sub(r'(?<!const )floating_block \*get_block_address',
               'const floating_block *get_block_address', t)
    t = re.sub(r'inline floating_block \*\s*\n(\s*vertex_partitioning::get_block_address)',
               r'inline const floating_block *\n\1', t)
    t = re.sub(r'(?<!const )floating_block \*block_pointer = &\(blocks\[block_id\]\);',
               'const floating_block *block_pointer = &(blocks[block_id]);', t)
    f.write_text(t)

f = d / 'lib/partition/onepass_partitioning/vertex_partitioning.cpp'
if f.exists():
    t = f.read_text()
    t = t.replace('absl::flat_hash_map', 'std::unordered_map')
    f.write_text(t)

# --- random_functions.h: constexpr -> const for large array (clang limit) ---
f = d / 'lib/tools/random_functions.h'
if f.exists():
    t = f.read_text()
    t = t.replace('constexpr ConstRandArray rand_nums',
                   'const ConstRandArray rand_nums')
    f.write_text(t)

# --- buffered_map.h: fix const-correctness (clang errors) ---
f = d / 'lib/data_structure/buffered_map.h'
if f.exists():
    t = f.read_text()
    # build() is const but calls non-const redo() - remove const from build()
    t = t.replace('void build() const {', 'void build() {')
    # comparison operators: add const qualifier (idempotent via negative lookbehind)
    t = re.sub(r'bool operator([<>=!]+)\(const minimal_min_heap<U>& obj\)(?! const) \{',
               r'bool operator\1(const minimal_min_heap<U>& obj) const {', t)
    # get_min(): ensure exactly one non-const + one const overload
    # Match get_min in both multi-line (original) and single-line (patched) format
    if 'const U& get_min() const' not in t:
        # Multi-line original: U& get_min() {\n\t\t\treturn (*this->fp)[0];\n\t\t}
        t = re.sub(
            r'(U& get_min\(\)\s*\{[^}]*\})',
            r'\1\n\t\tconst U& get_min() const { return (*this->fp)[0]; }',
            t, count=1)
    f.write_text(t)
]=])
execute_process(
    COMMAND python3 "${CMAKE_CURRENT_BINARY_DIR}/_patch_clustre.py" "${CLUSTRE_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Patch: replace exit() with throw in CluStRE files ---
foreach(_CLUSTRE_PATCH_FILE
    ${CLUSTRE_DIR}/lib/tools/quality_metrics.cpp
    ${CLUSTRE_DIR}/lib/tools/random_functions.cpp
)
    if(EXISTS ${_CLUSTRE_PATCH_FILE})
        file(READ ${_CLUSTRE_PATCH_FILE} _SRC)
        string(FIND "${_SRC}" "#include <stdexcept>" _HAS_STDEXCEPT)
        if(_HAS_STDEXCEPT EQUAL -1)
            string(REPLACE "#include" "#include <stdexcept>\n#include" _SRC "${_SRC}")
        endif()
        string(REPLACE "exit(0)" "throw std::runtime_error(\"CluStRE error (see stderr)\")" _SRC "${_SRC}")
        string(REPLACE "exit(1)" "throw std::runtime_error(\"CluStRE error (see stderr)\")" _SRC "${_SRC}")
        file(WRITE ${_CLUSTRE_PATCH_FILE} "${_SRC}")
    endif()
endforeach()

set(CLUSTRE_SOURCES
    ${CLUSTRE_DIR}/lib/partition/onepass_partitioning/vertex_partitioning.cpp
    ${CLUSTRE_DIR}/lib/partition/onepass_partitioning/floating_block.cpp
    ${CLUSTRE_DIR}/lib/partition/onepass_partitioning/leiden.cpp
    ${CLUSTRE_DIR}/lib/partition/onepass_partitioning/modularity.cpp
    ${CLUSTRE_DIR}/lib/tools/quality_metrics.cpp
    ${CLUSTRE_DIR}/lib/tools/random_functions.cpp
)

add_library(clustre_static STATIC ${CLUSTRE_SOURCES})
target_include_directories(clustre_static PUBLIC
    ${CLUSTRE_DIR}
    ${CLUSTRE_DIR}/app
    ${CLUSTRE_DIR}/lib
    ${CLUSTRE_DIR}/lib/partition
    ${CLUSTRE_DIR}/lib/tools
    ${CLUSTRE_DIR}/lib/data_structure
    ${CLUSTRE_DIR}/lib/data_structure/priority_queues
    ${CLUSTRE_DIR}/extern/robin-hood-hashing/src/include
)
target_compile_definitions(clustre_static PUBLIC
    -DMODE64BITEDGES
    -DPOINTER64=1
    -DNDEBUG
)
target_compile_options(clustre_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(clustre_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# FREIGHT  (Fast stREamInG Hypergraph parTitioning)
# =============================================================================
set(FREIGHT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/FREIGHT/code_for_hypergraphs)

# --- Patch definitions.h: remove x86intrin.h (ARM compatibility) ---
file(READ ${FREIGHT_DIR}/lib/definitions.h _SRC)
string(REPLACE "#include <x86intrin.h>" "// #include <x86intrin.h>  // patched out for portability" _SRC "${_SRC}")
file(WRITE ${FREIGHT_DIR}/lib/definitions.h "${_SRC}")

# --- Patch random_functions.h: constexpr array too large for some compilers ---
file(READ ${FREIGHT_DIR}/lib/tools/random_functions.h _SRC)
string(REPLACE "constexpr ConstRandArray rand_nums = ConstRandArray<TAM>();"
               "const ConstRandArray rand_nums = ConstRandArray<TAM>();" _SRC "${_SRC}")
file(WRITE ${FREIGHT_DIR}/lib/tools/random_functions.h "${_SRC}")

# --- Patch buffered_map.h: const-correctness (build() calls non-const redo(),
#     get_min() called on const refs in comparison operators) ---
file(READ ${FREIGHT_DIR}/lib/data_structure/buffered_map.h _SRC)
string(REPLACE "void build() const {" "void build() {" _SRC "${_SRC}")
string(REPLACE "U& get_min() {" "U& get_min() const {" _SRC "${_SRC}")
file(WRITE ${FREIGHT_DIR}/lib/data_structure/buffered_map.h "${_SRC}")

# --- Patch vertex_partitioning.h: const-correctness (get_block_address) ---
file(READ ${FREIGHT_DIR}/lib/partition/onepass_partitioning/vertex_partitioning.h _SRC)
string(REPLACE "floating_block* get_block_address(PartitionID block_id) const;"
               "floating_block* get_block_address(PartitionID block_id);" _SRC "${_SRC}")
string(REPLACE "inline floating_block* vertex_partitioning::get_block_address(PartitionID block_id) const {"
               "inline floating_block* vertex_partitioning::get_block_address(PartitionID block_id) {" _SRC "${_SRC}")
file(WRITE ${FREIGHT_DIR}/lib/partition/onepass_partitioning/vertex_partitioning.h "${_SRC}")

set(FREIGHT_SOURCES
    ${FREIGHT_DIR}/lib/partition/onepass_partitioning/vertex_partitioning.cpp
    ${FREIGHT_DIR}/lib/partition/onepass_partitioning/floating_block.cpp
    ${FREIGHT_DIR}/lib/partition/onepass_partitioning/fennel.cpp
    ${FREIGHT_DIR}/lib/partition/onepass_partitioning/fennel_approx_sqrt.cpp
    ${FREIGHT_DIR}/lib/partition/onepass_partitioning/ldg.cpp
    ${FREIGHT_DIR}/lib/tools/random_functions.cpp
    ${FREIGHT_DIR}/lib/tools/quality_metrics.cpp
    ${FREIGHT_DIR}/lib/mapping/hierarchy_operations.cpp
)

add_library(freight_static STATIC ${FREIGHT_SOURCES})
target_include_directories(freight_static PUBLIC
    ${FREIGHT_DIR}/lib
    ${FREIGHT_DIR}/lib/partition
    ${FREIGHT_DIR}/lib/partition/onepass_partitioning
    ${FREIGHT_DIR}/lib/tools
    ${FREIGHT_DIR}/lib/data_structure
    ${FREIGHT_DIR}/lib/io
    ${FREIGHT_DIR}/lib/mapping
    ${FREIGHT_DIR}/extern
)
target_compile_definitions(freight_static PUBLIC
    -DMODE64BITEDGES
    -DMODE_NETLIST
    -DMODE_FREIGHT
    -DNDEBUG
)
target_compile_options(freight_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(freight_static PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(freight_static PUBLIC Threads::Threads)

pybind11_add_module(_freight bindings/freight_binding.cpp)
target_link_libraries(_freight PRIVATE freight_static)
install(TARGETS _freight DESTINATION chszlablib)

# =============================================================================
# HyperMIS  (Maximum Independent Set on Hypergraphs)
# =============================================================================
set(HYPERMIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HyperMIS)

# --- hypergraph.cpp: fgetc_unlocked is glibc-only, use getc on other platforms ---
file(READ ${HYPERMIS_DIR}/src/hypergraph.cpp _SRC)
string(REPLACE "fgetc_unlocked" "getc" _SRC "${_SRC}")
file(WRITE ${HYPERMIS_DIR}/src/hypergraph.cpp "${_SRC}")

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
)
target_compile_options(hypermis_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -O2>)


# =============================================================================
# DynDeltaOrientation (Dynamic Edge Orientation — exact algorithms)
# =============================================================================
set(DYN_ORIENTATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/DynDeltaOrientation)

set(DYN_ORIENTATION_SOURCES
    ${DYN_ORIENTATION_DIR}/lib/data_structure/dyn_graph_access.cpp
    # NOTE: BucketPriorityQueue.cpp excluded (only used by ILP, conflicts with forced cstdint include)
    ${DYN_ORIENTATION_DIR}/lib/io/graph_io.cpp
    ${DYN_ORIENTATION_DIR}/lib/tools/timer.cpp
    ${DYN_ORIENTATION_DIR}/lib/tools/random_functions.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_BFSCS.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_BrodalFagerbergCS.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_KFlipsCS.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_MaxDecending.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_Naive.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_RWalkCS.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_impro_opt.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_improved_opt.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_improved_opt_dfs.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_naive_opt.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_strong_opt.cpp
    ${DYN_ORIENTATION_DIR}/lib/algorithms/dyn_edge_orientation_strong_opt_dfs.cpp
    # NOTE: dyn_edge_orientation_ILP.cpp and ilp.cpp excluded (require Gurobi)
)

add_library(dyn_orientation_static STATIC ${DYN_ORIENTATION_SOURCES})
target_include_directories(dyn_orientation_static PUBLIC
    ${DYN_ORIENTATION_DIR}/lib
    ${DYN_ORIENTATION_DIR}/lib/algorithms
    ${DYN_ORIENTATION_DIR}/lib/data_structure
    ${DYN_ORIENTATION_DIR}/lib/io
    ${DYN_ORIENTATION_DIR}/lib/tools
    ${DYN_ORIENTATION_DIR}/extern/sparsehash/src
)
# Do NOT define USEILP — algorithm_factory.h guards ILP behind #ifdef USEILP
target_compile_options(dyn_orientation_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive -O2>)
target_compile_definitions(dyn_orientation_static PRIVATE NDEBUG DELTAORI_IS_LOGGING=false)


# =============================================================================
# DynMatch (Dynamic Graph Matching)
# =============================================================================
set(DYN_MATCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/DynMatch)

set(DYN_MATCH_SOURCES
    ${DYN_MATCH_DIR}/lib/data_structure/dyn_graph_access.cpp
    ${DYN_MATCH_DIR}/lib/data_structure/graph_access.cpp
    ${DYN_MATCH_DIR}/lib/io/graph_io.cpp
    ${DYN_MATCH_DIR}/lib/tools/quality_metrics.cpp
    ${DYN_MATCH_DIR}/lib/tools/random_functions.cpp
    ${DYN_MATCH_DIR}/lib/tools/timer.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/baswanaguptasen_dyn_matching.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/blossom_dyn_matching.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/blossom_dyn_matching_naive.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/dyn_matching.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/naive_dyn_matching.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/neimansolomon_dyn_matching.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/rw_dyn_matching.cpp
    ${DYN_MATCH_DIR}/lib/algorithms/static_blossom.cpp
)

add_library(dyn_match_static STATIC ${DYN_MATCH_SOURCES})
target_include_directories(dyn_match_static PUBLIC
    ${DYN_MATCH_DIR}
    ${DYN_MATCH_DIR}/lib
    ${DYN_MATCH_DIR}/lib/algorithms
    ${DYN_MATCH_DIR}/lib/data_structure
    ${DYN_MATCH_DIR}/lib/io
    ${DYN_MATCH_DIR}/lib/tools
    ${DYN_MATCH_DIR}/extern/sparsehash/src
    ${DYN_MATCH_DIR}/extern
    ${DYN_MATCH_DIR}/extern/maxmatch_MV
)
target_compile_options(dyn_match_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive -O2>)
target_compile_definitions(dyn_match_static PRIVATE NDEBUG)


# =============================================================================
# DynWMIS (Dynamic Weighted Maximum Independent Set)
# =============================================================================
set(DYN_WMIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/DynWMIS)
set(DYN_WMIS_KAMIS_DIR ${DYN_WMIS_DIR}/extern/wmis)
set(DYN_WMIS_KAHIP_DIR ${DYN_WMIS_KAMIS_DIR}/extern/KaHIP)

# --- KaHIP sources (bundled fork inside DynWMIS, NO MPI files) ---
set(DYN_WMIS_KAHIP_SOURCES
    ${DYN_WMIS_KAHIP_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/algorithms/strongly_connected_components.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/algorithms/topological_sort.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/algorithms/push_relabel.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/algorithms/cycle_search.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/io/graph_io.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/tools/quality_metrics.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/tools/random_functions.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/tools/graph_extractor.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/tools/misc.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/tools/partition_snapshooter.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/graph_partitioner.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/coarsening.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/contraction.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/matching/matching.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/matching/random_matching.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/matching/gpa/path.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/clustering/node_ordering.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/initial_partitioning/bipartition.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/initial_partitioning/initial_node_separator.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/uncoarsening.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/separator/area_bfs.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/cut_flow_problem_solver.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/node_separators/greedy_ns_local_search.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/node_separators/fm_ns_local_search.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/node_separators/localized_fm_ns_local_search.cpp
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    # NOTE: Excluded MPI-dependent files: mpi_tools.cpp, graph_communication.cpp, parallel_mh/*
)

# --- WMIS (KaMIS MIS solvers) ---
set(DYN_WMIS_KAMIS_SOURCES
    ${DYN_WMIS_KAMIS_DIR}/lib/tools/mis_log.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/hils/Solution.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/ils/ils.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/ils/local_search.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/initial_mis/greedy_mis.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/initial_mis/greedy_vertex.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/initial_mis/random_mis.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/initial_mis/initial_mis.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/data_structure/mis_permutation.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/data_structure/candidate_list.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/data_structure/operation_log.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/data_structure/priority_queues/bucket_array.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel/branch_and_reduce_algorithm.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel/reductions.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel/mwis_finder.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel/original_struction.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel/extended_struction.cpp
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel/key_functions.cpp
)

# --- OnlineMIS ---
set(DYN_WMIS_ONLINEMIS_SOURCES
    ${DYN_WMIS_DIR}/extern/onlinemis/ils/local_search_online.cpp
    ${DYN_WMIS_DIR}/extern/onlinemis/ils/online_ils.cpp
    ${DYN_WMIS_DIR}/extern/onlinemis/tools/mis_log.cpp
    ${DYN_WMIS_DIR}/extern/onlinemis/initial_mis/greedy_mis_online.cpp
    ${DYN_WMIS_DIR}/extern/onlinemis/data_structure/candidate_list.cpp
    ${DYN_WMIS_DIR}/extern/onlinemis/data_structure/mis_permutation_online.cpp
    ${DYN_WMIS_DIR}/extern/onlinemis/data_structure/priority_queues/bucket_array.cpp
)

# --- DynWMIS core ---
set(DYN_WMIS_CORE_SOURCES
    ${DYN_WMIS_DIR}/lib/data_structure/dyn_graph_access.cpp
    ${DYN_WMIS_DIR}/lib/io/mwis_graph_io.cpp
    ${DYN_WMIS_DIR}/lib/tools/timer.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/dyn_wmis_simple.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/dyn_wmis_greedy.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/dyn_wmis_greedy_deg.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/dyn_wmis_bfs.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/dyn_wmis_static.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/solve_static_wmis.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/mis_branch_and_reduce_algorithm.cpp
    ${DYN_WMIS_DIR}/lib/algorithms/modified.cpp
)

add_library(dyn_wmis_static STATIC
    ${DYN_WMIS_CORE_SOURCES}
    ${DYN_WMIS_KAMIS_SOURCES}
    ${DYN_WMIS_KAHIP_SOURCES}
    ${DYN_WMIS_ONLINEMIS_SOURCES}
)
target_include_directories(dyn_wmis_static PUBLIC
    # DynWMIS core
    ${DYN_WMIS_DIR}/lib
    ${DYN_WMIS_DIR}/lib/algorithms
    ${DYN_WMIS_DIR}/lib/data_structure
    ${DYN_WMIS_DIR}/lib/io
    ${DYN_WMIS_DIR}/lib/tools
    ${DYN_WMIS_DIR}/extern/sparsehash/src
    # WMIS (KaMIS fork)
    ${DYN_WMIS_KAMIS_DIR}/app
    ${DYN_WMIS_KAMIS_DIR}/lib
    ${DYN_WMIS_KAMIS_DIR}/lib/io
    ${DYN_WMIS_KAMIS_DIR}/lib/mis
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/hils
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/ils
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/initial_mis
    ${DYN_WMIS_KAMIS_DIR}/lib/mis/kernel
    ${DYN_WMIS_KAMIS_DIR}/lib/partition
    ${DYN_WMIS_KAMIS_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${DYN_WMIS_KAMIS_DIR}/lib/tools
    ${DYN_WMIS_KAMIS_DIR}/lib/data_structure
    ${DYN_WMIS_KAMIS_DIR}/lib/data_structure/priority_queues
    ${DYN_WMIS_KAMIS_DIR}/extern
    # KaHIP fork
    ${DYN_WMIS_KAHIP_DIR}
    ${DYN_WMIS_KAHIP_DIR}/lib
    ${DYN_WMIS_KAHIP_DIR}/lib/io
    ${DYN_WMIS_KAHIP_DIR}/lib/partition
    ${DYN_WMIS_KAHIP_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${DYN_WMIS_KAHIP_DIR}/lib/tools
    ${DYN_WMIS_KAHIP_DIR}/lib/data_structure
    ${DYN_WMIS_KAHIP_DIR}/lib/data_structure/matrix
    ${DYN_WMIS_KAHIP_DIR}/lib/data_structure/priority_queues
    # OnlineMIS
    ${DYN_WMIS_DIR}/extern/onlinemis
    ${DYN_WMIS_DIR}/extern/onlinemis/ils
    ${DYN_WMIS_DIR}/extern/onlinemis/initial_mis
    ${DYN_WMIS_DIR}/extern/onlinemis/data_structure
)
target_compile_options(dyn_wmis_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w -fpermissive -O2>)
target_compile_definitions(dyn_wmis_static PRIVATE NDEBUG)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dyn_wmis_static PUBLIC OpenMP::OpenMP_CXX)
endif()


# =============================================================================
# Pybind11 modules
# =============================================================================

# --- _kahip ---
pybind11_add_module(_kahip bindings/kahip_binding.cpp)
target_link_libraries(_kahip PRIVATE kahip_static)
install(TARGETS _kahip DESTINATION chszlablib)

# --- _kahipe (evolutionary/memetic partitioner) ---
pybind11_add_module(_kahipe bindings/kahipe_binding.cpp)
target_link_libraries(_kahipe PRIVATE kahipe_static)
install(TARGETS _kahipe DESTINATION chszlablib)

# --- VieCut: replace growt definitions.h on non-Linux ---
# growt is a Linux-only concurrent hash table (uses futex, cpu_set_t, SSE).
# On non-Linux, replace definitions.h with a portable shim that wraps
# std::unordered_map + std::mutex, providing the same uaGrow interface
# that VieCut's parallel contract_graph.h uses.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(WRITE "${VIECUT_DIR}/extlib/growt/data-structures/definitions.h"
"#ifndef DEFINITIONS_H\n\
#define DEFINITIONS_H\n\
// Portable growt replacement for non-Linux (macOS, etc.)\n\
// Provides the uaGrow<Hash> interface used by VieCut's contract_graph.h\n\
#include <unordered_map>\n\
#include <mutex>\n\
#include <utility>\n\
#include <cstdint>\n\
#include <cstddef>\n\
namespace growt {\n\
template<class HashFct = std::hash<uint64_t>>\n\
class uaGrow {\n\
public:\n\
    explicit uaGrow(size_t = 0) {}\n\
    class Handle {\n\
    public:\n\
        Handle(std::unordered_map<uint64_t,size_t,HashFct>& m, std::mutex& mu)\n\
            : map_(m), mu_(mu) {}\n\
        using iterator = typename std::unordered_map<uint64_t,size_t,HashFct>::iterator;\n\
        iterator find(uint64_t key) {\n\
            std::lock_guard<std::mutex> lock(mu_);\n\
            return map_.find(key);\n\
        }\n\
        template<class F, class... Args>\n\
        std::pair<size_t, bool> insert_or_update(\n\
                uint64_t key, size_t val, F f, Args&&... args) {\n\
            std::lock_guard<std::mutex> lock(mu_);\n\
            auto it = map_.find(key);\n\
            if (it != map_.end()) {\n\
                f(it->second, std::forward<Args>(args)...);\n\
                return {it->second, false};\n\
            }\n\
            map_.emplace(key, val);\n\
            return {val, true};\n\
        }\n\
    private:\n\
        std::unordered_map<uint64_t,size_t,HashFct>& map_;\n\
        std::mutex& mu_;\n\
    };\n\
    Handle get_handle() { return Handle(map_, mu_); }\n\
private:\n\
    std::unordered_map<uint64_t,size_t,HashFct> map_;\n\
    std::mutex mu_;\n\
};\n\
}\n\
#endif\n"
    )
endif()

# --- _viecut ---
pybind11_add_module(_viecut bindings/viecut_binding.cpp)
target_include_directories(_viecut PRIVATE
    ${VIECUT_DIR}/lib
    ${VIECUT_DIR}/app
    ${VIECUT_DIR}/extlib/growt
    ${VIECUT_DIR}/extlib/tlx
)
target_link_libraries(_viecut PRIVATE tlx)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_viecut PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_viecut PRIVATE PARALLEL)
endif()
# Parallel VieCut uses 128-bit atomics (__sync_bool_compare_and_swap_16).
# On x86_64, tell GCC to emit cmpxchg16b inline; also link libatomic as fallback.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    target_compile_options(_viecut PRIVATE -mcx16)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_viecut PRIVATE "LINKER:--no-as-needed")
    target_link_libraries(_viecut PRIVATE atomic)
endif()
install(TARGETS _viecut DESTINATION chszlablib)

# --- _vieclus ---
pybind11_add_module(_vieclus bindings/vieclus_binding.cpp)
target_link_libraries(_vieclus PRIVATE vieclus_static)
install(TARGETS _vieclus DESTINATION chszlablib)

# --- _chils ---
pybind11_add_module(_chils bindings/chils_binding.cpp)
target_link_libraries(_chils PRIVATE chils_static)
install(TARGETS _chils DESTINATION chszlablib)

# --- _kamis (ReduMIS + OnlineMIS) ---
pybind11_add_module(_kamis bindings/kamis_binding.cpp)
target_link_libraries(_kamis PRIVATE kamis_redumis_static)
install(TARGETS _kamis DESTINATION chszlablib)

# --- _kamis_wmis (Branch & Reduce) ---
pybind11_add_module(_kamis_wmis bindings/kamis_wmis_binding.cpp)
target_link_libraries(_kamis_wmis PRIVATE kamis_wmis_static)
install(TARGETS _kamis_wmis DESTINATION chszlablib)

# --- _kamis_mmwis (MMWIS) ---
pybind11_add_module(_kamis_mmwis bindings/kamis_mmwis_binding.cpp)
target_link_libraries(_kamis_mmwis PRIVATE kamis_mmwis_static)
install(TARGETS _kamis_mmwis DESTINATION chszlablib)

# --- _learnandreduce (GNN-guided MWIS kernelization) ---
pybind11_add_module(_learnandreduce bindings/learnandreduce_binding.cpp)
target_link_libraries(_learnandreduce PRIVATE learnandreduce_static)
install(TARGETS _learnandreduce DESTINATION chszlablib)

# --- _red2pack (Maximum 2-Packing Set) ---
pybind11_add_module(_red2pack bindings/red2pack_binding.cpp)
target_link_libraries(_red2pack PRIVATE red2pack_static)
target_include_directories(_red2pack PRIVATE ${R2P_DIR}/app)
install(TARGETS _red2pack DESTINATION chszlablib)

# --- _scc (Correlation Clustering) ---
pybind11_add_module(_scc bindings/scc_binding.cpp)
target_link_libraries(_scc PRIVATE scc_static)
install(TARGETS _scc DESTINATION chszlablib)

# --- _scc_evo (Evolutionary Correlation Clustering) ---
pybind11_add_module(_scc_evo bindings/scc_evo_binding.cpp)
target_link_libraries(_scc_evo PRIVATE scc_evo_static)
install(TARGETS _scc_evo DESTINATION chszlablib)

# --- _heiorient (Edge Orientation) ---
pybind11_add_module(_heiorient bindings/heiorient_binding.cpp)
target_link_libraries(_heiorient PRIVATE heiorient)
target_compile_options(_heiorient PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-c++11-narrowing>)
install(TARGETS _heiorient DESTINATION chszlablib)

# --- _maxcut (FPT Max-Cut) ---
pybind11_add_module(_maxcut bindings/maxcut_binding.cpp)
target_link_libraries(_maxcut PRIVATE fpt_maxcut_static)
install(TARGETS _maxcut DESTINATION chszlablib)

# --- _motif (Motif Clustering: SOCIAL + LMCHGP) ---
pybind11_add_module(_motif bindings/motif_binding.cpp)
target_link_libraries(_motif PRIVATE motif_static)
target_compile_definitions(_motif PRIVATE
    -DNDEBUG
    -DMODE64BITEDGES
    -DPOINTER64=1
    -DMODE_MOTIF_CLUSTERING_GRAPH
)
install(TARGETS _motif DESTINATION chszlablib)

# --- _heistream (Streaming Graph Partitioner) ---
pybind11_add_module(_heistream bindings/heistream_binding.cpp)
target_link_libraries(_heistream PRIVATE heistream_static)
install(TARGETS _heistream DESTINATION chszlablib)

# --- _clustre (Streaming Graph Clustering) ---
pybind11_add_module(_clustre bindings/clustre_binding.cpp)
target_link_libraries(_clustre PRIVATE clustre_static)
target_compile_options(_clustre PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fpermissive>)
install(TARGETS _clustre DESTINATION chszlablib)

# --- _hypermis (HyperMIS reductions, always built) ---
pybind11_add_module(_hypermis bindings/hypermis_binding.cpp)
target_link_libraries(_hypermis PRIVATE hypermis_static)
install(TARGETS _hypermis DESTINATION chszlablib)

# --- _io (Fast METIS/hMETIS file parser) ---
pybind11_add_module(_io bindings/io_binding.cpp)
install(TARGETS _io DESTINATION chszlablib)

# =============================================================================
# HeiCut  (Exact Minimum Cut on Hypergraphs)
# =============================================================================
set(HEICUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiCut)
set(MTKAHYPAR_DIR ${HEICUT_DIR}/extern/mt-kahypar)
set(MTKAHYPAR_BUILD_DIR ${MTKAHYPAR_DIR}/build)
set(MTKAHYPAR_LIB_DIR ${HEICUT_DIR}/extern/mt-kahypar-library)
# TBB headers: check build_mtkahypar.sh install dir first, then mt-kahypar build dir
if(EXISTS ${MTKAHYPAR_LIB_DIR}/tbb_include)
    set(TBB_INCLUDE_DIR ${MTKAHYPAR_LIB_DIR}/tbb_include)
    set(TBB_LIB_SEARCH_DIR ${MTKAHYPAR_LIB_DIR}/tbb_lib/intel64/gcc4.8)
elseif(EXISTS ${MTKAHYPAR_BUILD_DIR}/external_tools/tbb/include)
    set(TBB_INCLUDE_DIR ${MTKAHYPAR_BUILD_DIR}/external_tools/tbb/include)
    set(TBB_LIB_SEARCH_DIR ${MTKAHYPAR_BUILD_DIR}/external_tools/tbb/lib/intel64/gcc4.8)
else()
    # System TBB (macOS Homebrew: /opt/homebrew/include)
    find_path(TBB_INCLUDE_DIR tbb/parallel_for.h)
    set(TBB_LIB_SEARCH_DIR "")
endif()

# --- HeiCut static library ---
add_library(heicut_static STATIC
    ${HEICUT_DIR}/lib/solvers/ilp.cpp
    ${HEICUT_DIR}/lib/solvers/kernelizer.cpp
    ${HEICUT_DIR}/lib/solvers/submodular.cpp
    ${HEICUT_DIR}/lib/coarsening/label_propagation.cpp
    ${HEICUT_DIR}/lib/decomposition/core_decomposition.cpp
    ${HEICUT_DIR}/lib/trimmer/trimmer.cpp
    ${HEICUT_DIR}/lib/parse_parameters/parse_parameters.cpp
    ${HEICUT_DIR}/lib/utils/random.cpp
    ${HEICUT_DIR}/extern/argtable3-3.2.2/argtable3.c
)

target_include_directories(heicut_static BEFORE PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/gurobi          # Gurobi C++ shim (MUST be first)
)
target_include_directories(heicut_static PUBLIC
    ${HEICUT_DIR}                                      # for "lib/..." includes
    ${HEICUT_DIR}/extern/mt-kahypar-library            # libmtkahypar.h
    ${MTKAHYPAR_DIR}                                   # mt-kahypar/... internal headers
    ${MTKAHYPAR_DIR}/include                           # mt-kahypar public includes
    ${MTKAHYPAR_DIR}/external_tools/kahypar-shared-resources # kahypar-resources/...
    ${MTKAHYPAR_DIR}/external_tools/growt              # growt hash tables
    ${HEICUT_DIR}/extern                               # for "kahip/..." includes
    ${TBB_INCLUDE_DIR}                                  # TBB headers
    ${HEICUT_DIR}/extern/argtable3-3.2.2               # argtable3.h for parse_parameters
    ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiStream/extern/sparsehash/src  # sparsehash for pruner
)

# Add Boost program_options include from mt-kahypar's downloaded copy
if(EXISTS ${MTKAHYPAR_LIB_DIR}/boost_include/boost)
    target_include_directories(heicut_static PUBLIC ${MTKAHYPAR_LIB_DIR}/boost_include)
elseif(EXISTS ${MTKAHYPAR_BUILD_DIR}/external_tools/boost/boost)
    target_include_directories(heicut_static PUBLIC ${MTKAHYPAR_BUILD_DIR}/external_tools/boost)
endif()

target_compile_features(heicut_static PUBLIC cxx_std_17)
target_compile_options(heicut_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
target_compile_definitions(heicut_static PRIVATE NDEBUG)

# Gurobi shim needs pybind11 + Python headers (pybind11::embed requires
# libpython which isn't available in manylinux). pybind11::pybind11 provides
# headers + Python include without linking libpython.
target_link_libraries(heicut_static PUBLIC pybind11::pybind11)

# Link TBB (pre-built from Mt-KaHyPar build)
find_library(TBB_LIB tbb PATHS ${TBB_LIB_SEARCH_DIR} NO_DEFAULT_PATH)
if(NOT TBB_LIB)
    find_library(TBB_LIB tbb)  # fall back to system TBB (macOS Homebrew)
endif()
if(TBB_LIB)
    target_link_libraries(heicut_static PUBLIC ${TBB_LIB})
endif()

# Link libmtkahypar.so (pre-built)
find_library(MTKAHYPAR_LIB mtkahypar PATHS ${MTKAHYPAR_LIB_DIR} ${MTKAHYPAR_BUILD_DIR}/lib NO_DEFAULT_PATH)
if(MTKAHYPAR_LIB)
    target_link_libraries(heicut_static PUBLIC ${MTKAHYPAR_LIB})
endif()

# Link hwloc (required by Mt-KaHyPar)
find_library(HWLOC_LIB hwloc)
if(HWLOC_LIB)
    target_link_libraries(heicut_static PUBLIC ${HWLOC_LIB})
endif()

find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(heicut_static PUBLIC Threads::Threads)
endif()

# --- _heicut pybind11 module ---
pybind11_add_module(_heicut bindings/heicut_binding.cpp)
target_link_libraries(_heicut PRIVATE heicut_static)
# Embed RPATH so libmtkahypar.so and libtbb.so can be found at runtime.
# BUILD_RPATH: absolute paths for the build tree (local development).
# INSTALL_RPATH: $ORIGIN so auditwheel/delocate can bundle the .so/.dylib
# into the wheel at install time.
set_target_properties(_heicut PROPERTIES
    BUILD_RPATH "${MTKAHYPAR_BUILD_DIR}/lib;${TBB_DIR}/lib/intel64/gcc4.8"
    INSTALL_RPATH "$ORIGIN"
)
install(TARGETS _heicut DESTINATION chszlablib)

# =============================================================================
# SharedMap  (Hierarchical Process Mapping)
# =============================================================================
set(SHAREDMAP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/SharedMap)

# Use SharedMap's own Mt-KaHyPar v1.5.3 build (matches CLI exactly).
# On macOS, run scripts/build_sharedmap_mtkahypar.sh to build from source.
set(SHAREDMAP_MTKAHYPAR_DIR ${SHAREDMAP_DIR}/extern/local/mt-kahypar)
set(SHAREDMAP_MTKAHYPAR_INCLUDE ${SHAREDMAP_MTKAHYPAR_DIR}/include)
# Platform-aware library extension
if(APPLE)
    set(_SM_LIB_EXT "dylib")
else()
    set(_SM_LIB_EXT "so")
endif()
# Try lib/ first, fall back to lib64/
# On macOS, the library is renamed to libmtkahypar_sm.dylib to avoid
# collision with HeiCut's libmtkahypar.dylib.
if(APPLE)
    set(_SM_LIB_NAME "mtkahypar_sm")
else()
    set(_SM_LIB_NAME "mtkahypar")
endif()
if(EXISTS "${SHAREDMAP_MTKAHYPAR_DIR}/lib/lib${_SM_LIB_NAME}.${_SM_LIB_EXT}")
    set(SHAREDMAP_MTKAHYPAR_LIB_DIR ${SHAREDMAP_MTKAHYPAR_DIR}/lib)
elseif(EXISTS "${SHAREDMAP_MTKAHYPAR_DIR}/lib64/lib${_SM_LIB_NAME}.${_SM_LIB_EXT}")
    set(SHAREDMAP_MTKAHYPAR_LIB_DIR ${SHAREDMAP_MTKAHYPAR_DIR}/lib64)
else()
    set(SHAREDMAP_MTKAHYPAR_LIB_DIR "")
endif()

# Only build SharedMap when Mt-KaHyPar headers are available.
# Run scripts/build_sharedmap_mtkahypar.sh first on fresh checkouts.
if(EXISTS "${SHAREDMAP_MTKAHYPAR_INCLUDE}/mtkahypar.h")

# Patch SharedMap layer.h for macOS: size_t (unsigned long) vs u64 (unsigned long long)
# type mismatch in std::min/max calls that AppleClang treats as hard errors.
if(APPLE)
    set(_LAYER_H "${SHAREDMAP_DIR}/src/partitioning/layer.h")
    file(READ "${_LAYER_H}" _LAYER_CONTENT)
    string(FIND "${_LAYER_CONTENT}" "(u64) layer.size()" _ALREADY_PATCHED)
    if(_ALREADY_PATCHED EQUAL -1)
        string(REPLACE
            "std::min(layer.size(), n_threads)"
            "std::min((u64) layer.size(), n_threads)"
            _LAYER_CONTENT "${_LAYER_CONTENT}")
        string(REPLACE
            "(n_threads / layer.size()))"
            "(u64)(n_threads / layer.size()))"
            _LAYER_CONTENT "${_LAYER_CONTENT}")
        file(WRITE "${_LAYER_H}" "${_LAYER_CONTENT}")
        message(STATUS "SharedMap: patched layer.h for macOS size_t/u64 compatibility")
    endif()
endif()

add_library(sharedmap_static STATIC
    ${SHAREDMAP_DIR}/lib/libsharedmap.cpp
    ${SHAREDMAP_DIR}/src/utility/build_anchor.cpp
)

# Use SharedMap's own v1.5.3 headers (no shims needed — native API match).
target_include_directories(sharedmap_static PUBLIC
    ${SHAREDMAP_DIR}                                       # for "include/...", "src/..." includes
    ${SHAREDMAP_MTKAHYPAR_INCLUDE}                         # mtkahypar.h, mtkahypartypes.h (v1.5.3)
    ${TBB_INCLUDE_DIR}                                     # TBB headers
    ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/KaHIP/interface  # kaHIP_interface.h
)

target_compile_features(sharedmap_static PUBLIC cxx_std_17)
target_compile_options(sharedmap_static PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
target_compile_definitions(sharedmap_static PRIVATE NDEBUG)

target_link_libraries(sharedmap_static PUBLIC kahip_static)
if(SHAREDMAP_MTKAHYPAR_LIB_DIR)
    # Link against SharedMap's own Mt-KaHyPar v1.5.3 (same as CLI)
    # On macOS, uses renamed libmtkahypar_sm.dylib to avoid HeiCut collision.
    find_library(SHAREDMAP_MTKAHYPAR_LIB ${_SM_LIB_NAME}
        PATHS ${SHAREDMAP_MTKAHYPAR_LIB_DIR} NO_DEFAULT_PATH)
    target_link_libraries(sharedmap_static PUBLIC ${SHAREDMAP_MTKAHYPAR_LIB})
elseif(MTKAHYPAR_LIB)
    target_link_libraries(sharedmap_static PUBLIC ${MTKAHYPAR_LIB})
endif()
if(TBB_LIB)
    target_link_libraries(sharedmap_static PUBLIC ${TBB_LIB})
endif()
if(HWLOC_LIB)
    target_link_libraries(sharedmap_static PUBLIC ${HWLOC_LIB})
endif()
if(Threads_FOUND)
    target_link_libraries(sharedmap_static PUBLIC Threads::Threads)
endif()

# --- _sharedmap pybind11 module ---
pybind11_add_module(_sharedmap bindings/sharedmap_binding.cpp)
target_link_libraries(_sharedmap PRIVATE sharedmap_static)
set_target_properties(_sharedmap PROPERTIES
    BUILD_RPATH "${SHAREDMAP_MTKAHYPAR_LIB_DIR};${TBB_DIR}/lib/intel64/gcc4.8"
    INSTALL_RPATH "$ORIGIN"
)
install(TARGETS _sharedmap DESTINATION chszlablib)

# Install SharedMap's Mt-KaHyPar into chszlablib/ for runtime loading.
if(SHAREDMAP_MTKAHYPAR_LIB_DIR)
    if(APPLE)
        # macOS: install renamed libmtkahypar_sm.dylib to avoid collision with HeiCut's
        file(GLOB _SM_MTKAHYPAR_DYLIBS "${SHAREDMAP_MTKAHYPAR_LIB_DIR}/libmtkahypar_sm.dylib*")
        if(_SM_MTKAHYPAR_DYLIBS)
            install(FILES ${_SM_MTKAHYPAR_DYLIBS} DESTINATION chszlablib)
        endif()
        # Set RPATH so _sharedmap.so finds libmtkahypar_sm.dylib next to itself
        install(CODE "
            execute_process(COMMAND install_name_tool -add_rpath @loader_path
                \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/chszlablib/_sharedmap\${CMAKE_SHARED_MODULE_SUFFIX}
                ERROR_QUIET)
        ")
    else()
        # Linux: install versioned .so (soname libmtkahypar.so.1) to avoid
        # conflict with HeiCut's unversioned libmtkahypar.so.
        install(FILES
            "${SHAREDMAP_MTKAHYPAR_LIB_DIR}/libmtkahypar.so.1.5.3"
            DESTINATION chszlablib)
        # Also install the soname symlink
        install(CODE "
            execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                libmtkahypar.so.1.5.3 \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/chszlablib/libmtkahypar.so.1)
        ")
        # Install bundled dependencies (required by Mt-KaHyPar v1.5.3)
        file(GLOB _SHAREDMAP_BOOST_LIBS "${SHAREDMAP_MTKAHYPAR_LIB_DIR}/libboost_program_options.so*")
        if(_SHAREDMAP_BOOST_LIBS)
            install(FILES ${_SHAREDMAP_BOOST_LIBS} DESTINATION chszlablib)
        endif()
        # Install bundled TBB (v1.5.3 needs newer TBB than system may have)
        file(GLOB _SHAREDMAP_TBB_LIBS
            "${SHAREDMAP_MTKAHYPAR_LIB_DIR}/libtbb.so*"
            "${SHAREDMAP_MTKAHYPAR_LIB_DIR}/libtbbmalloc.so*")
        if(_SHAREDMAP_TBB_LIBS)
            install(FILES ${_SHAREDMAP_TBB_LIBS} DESTINATION chszlablib)
        endif()
        # Patch libmtkahypar.so.1.5.3 RPATH so it finds bundled TBB/Boost
        find_program(PATCHELF_EXECUTABLE patchelf)
        if(PATCHELF_EXECUTABLE)
            install(CODE "
                execute_process(COMMAND ${PATCHELF_EXECUTABLE} --set-rpath \"\\$ORIGIN\"
                    \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/chszlablib/libmtkahypar.so.1.5.3)
            ")
        endif()
    endif()
endif()

else()
    message(STATUS "SharedMap: skipping (Mt-KaHyPar headers not found at ${SHAREDMAP_MTKAHYPAR_INCLUDE})")
endif() # EXISTS mtkahypar.h

# Install HeiCut's Mt-KaHyPar .so into chszlablib/ for _heicut runtime loading.
if(MTKAHYPAR_LIB)
    install(FILES "${MTKAHYPAR_LIB}" DESTINATION chszlablib)
endif()

# =============================================================================
# HeiHGM  (Hypergraph B-Matching — Static + Streaming)
# =============================================================================
set(HEIHGM_BMATCHING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiHGM_Bmatching)
set(HEIHGM_STREAMING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/HeiHGM_Streaming)

# --- Patch Abseil/Protobuf out of both repos at configure time ---
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/patch_heihgm.py ${HEIHGM_BMATCHING_DIR}
    RESULT_VARIABLE _patch_bm_rc
)
if(NOT _patch_bm_rc EQUAL 0)
    message(WARNING "HeiHGM Bmatching patching returned non-zero: ${_patch_bm_rc}")
endif()
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/patch_heihgm.py ${HEIHGM_STREAMING_DIR}
    RESULT_VARIABLE _patch_st_rc
)
if(NOT _patch_st_rc EQUAL 0)
    message(WARNING "HeiHGM Streaming patching returned non-zero: ${_patch_st_rc}")
endif()

# --- Bmatching: header-only (INTERFACE library) ---
add_library(bmatching_headers INTERFACE)
target_include_directories(bmatching_headers INTERFACE
    ${HEIHGM_BMATCHING_DIR}
    ${HEIHGM_BMATCHING_DIR}/others    # easylogging++.h, ds/hashing-config.h
)
target_compile_features(bmatching_headers INTERFACE cxx_std_17)

# --- _bmatching pybind11 module ---
pybind11_add_module(_bmatching bindings/bmatching_binding.cpp)
target_link_libraries(_bmatching PRIVATE bmatching_headers)
target_include_directories(_bmatching BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/gurobi          # Gurobi C++ shim (gurobipy backend)
)
target_compile_definitions(_bmatching PRIVATE USE_GUROBI_ENABLED)
target_compile_options(_bmatching PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
install(TARGETS _bmatching DESTINATION chszlablib)

# --- _streaming_bmatching pybind11 module ---
# Streaming algorithms are reimplemented directly in the binding (no external deps)
pybind11_add_module(_streaming_bmatching bindings/streaming_bmatching_binding.cpp)
target_compile_features(_streaming_bmatching PRIVATE cxx_std_17)
install(TARGETS _streaming_bmatching DESTINATION chszlablib)

# --- _dyn_orientation (Dynamic Edge Orientation — exact) ---
pybind11_add_module(_dyn_orientation bindings/dyn_orientation_binding.cpp)
target_link_libraries(_dyn_orientation PRIVATE dyn_orientation_static)
target_compile_options(_dyn_orientation PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
install(TARGETS _dyn_orientation DESTINATION chszlablib)

# --- _dyn_delta_approx (Dynamic Edge Orientation — approximate, header-only) ---
set(DYN_DELTA_APPROX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/DynDeltaApprox)
pybind11_add_module(_dyn_delta_approx bindings/dyn_delta_approx_binding.cpp)
target_include_directories(_dyn_delta_approx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/dyn_delta_approx_stubs
    ${DYN_DELTA_APPROX_DIR}
)
target_compile_options(_dyn_delta_approx PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
install(TARGETS _dyn_delta_approx DESTINATION chszlablib)

# --- _dyn_matching (Dynamic Graph Matching) ---
pybind11_add_module(_dyn_matching bindings/dyn_matching_binding.cpp)
target_link_libraries(_dyn_matching PRIVATE dyn_match_static)
target_compile_options(_dyn_matching PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
install(TARGETS _dyn_matching DESTINATION chszlablib)

# --- _dyn_wmis (Dynamic Weighted MIS) ---
pybind11_add_module(_dyn_wmis bindings/dyn_wmis_binding.cpp)
target_link_libraries(_dyn_wmis PRIVATE dyn_wmis_static)
target_compile_options(_dyn_wmis PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
install(TARGETS _dyn_wmis DESTINATION chszlablib)

