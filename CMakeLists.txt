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

if(OpenMP_C_FOUND)
    target_link_libraries(chils_static PUBLIC OpenMP::OpenMP_C)
    if(MSVC)
        # MSVC's default /openmp uses OpenMP 2.0 which rejects C99-style
        # for-loop init inside `#pragma omp for` (error C3015).
        # Override with /openmp:llvm which supports OpenMP 3.0+ and accepts it.
        target_compile_options(chils_static PRIVATE /openmp:llvm)
    endif()
endif()

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
file(READ ${_STRUCTION_DIR}/struction_reductions.cpp _SRC)
# Wrap each template instantiation in its own namespace block
string(REPLACE "template class iterative_struction<" "namespace struction { template class iterative_struction<" _SRC "${_SRC}")
string(REPLACE "template class blow_up_struction<"    "namespace struction { template class blow_up_struction<"    _SRC "${_SRC}")
# Close the namespace at end of each line (after the semicolon)
string(REGEX REPLACE "(namespace struction \\{ template class [^;]+;)" "\\1 }" _SRC "${_SRC}")
file(WRITE ${_STRUCTION_DIR}/struction_reductions.cpp "${_SRC}")

# --- extended_struction.cpp ---
file(READ ${_STRUCTION_DIR}/extended_struction.cpp _SRC)
string(REPLACE "template class extended_struction<" "namespace struction { template class extended_struction<" _SRC "${_SRC}")
string(REGEX REPLACE "(namespace struction \\{ template class [^;]+;)" "\\1 }" _SRC "${_SRC}")
file(WRITE ${_STRUCTION_DIR}/extended_struction.cpp "${_SRC}")

# --- original_struction.cpp ---
file(READ ${_STRUCTION_DIR}/original_struction.cpp _SRC)
string(REPLACE "template class original_struction<" "namespace struction { template class original_struction<" _SRC "${_SRC}")
string(REGEX REPLACE "(namespace struction \\{ template class [^;]+;)" "\\1 }" _SRC "${_SRC}")
file(WRITE ${_STRUCTION_DIR}/original_struction.cpp "${_SRC}")

# --- reduction_evolution.cpp: namespace + default argument redefinition ---
set(_MMWIS_EVO_DIR ${KAMIS_DIR}/mmwis/lib/mis/evolutionary)
file(READ ${_MMWIS_EVO_DIR}/reduction_evolution.cpp _SRC)
# Clang rejects default arg redefinition in definition when already in declaration
string(REPLACE "bool recursive = false," "bool recursive," _SRC "${_SRC}")
string(REPLACE
    "template class reduction_evolution<branch_and_reduce_algorithm>;"
    "namespace mmwis { template class reduction_evolution<branch_and_reduce_algorithm>; }"
    _SRC "${_SRC}")
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

# --- Patch: MSVC rejects narrowing size_t → unsigned int inside brace init ---
file(READ ${HEIORIENT_DIR}/edge_orientation/ds/smaller_graph.h _SRC)
string(REPLACE "IndexType{out_s.size() - 1}" "IndexType{static_cast<IT>(out_s.size() - 1)}" _SRC "${_SRC}")
file(WRITE ${HEIORIENT_DIR}/edge_orientation/ds/smaller_graph.h "${_SRC}")

# =============================================================================
# KaHIP Evolutionary (kaffpaE) — modified_kahip with pseudo-MPI shim
# =============================================================================
set(KAHIP_PARALLEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/KaHIP/parallel/modified_kahip)

add_library(kahipe_static STATIC
    ${KAHIP_PARALLEL_DIR}/interface/kaHIP_interface.cpp
    # Core KaHIP sources (modified_kahip's own copy)
    ${KAHIP_PARALLEL_DIR}/lib/data_structure/graph_hierarchy.cpp
    ${KAHIP_PARALLEL_DIR}/lib/algorithms/strongly_connected_components.cpp
    ${KAHIP_PARALLEL_DIR}/lib/algorithms/topological_sort.cpp
    ${KAHIP_PARALLEL_DIR}/lib/algorithms/cycle_search.cpp
    ${KAHIP_PARALLEL_DIR}/lib/io/graph_io.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/quality_metrics.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/random_functions.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/graph_extractor.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/misc.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/partition_snapshooter.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/graph_partitioner.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/w_cycles/wcycle_partitioner.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/coarsening.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/contraction.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/edge_rating/edge_ratings.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/matching/matching.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/matching/random_matching.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/matching/gpa/path.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/matching/gpa/gpa_matching.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/matching/gpa/path_set.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/clustering/node_ordering.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/initial_partitioning/initial_partitioning.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/initial_partitioning/initial_partitioner.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/initial_partitioning/initial_partition_bipartition.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/initial_partitioning/initial_refinement/initial_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/initial_partitioning/bipartition.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/uncoarsening.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/separator/vertex_separator_algorithm.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/separator/vertex_separator_flow_solver.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/greedy_neg_cycle.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/problem_factory.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/augmented_Qgraph_fabric.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/advanced_models.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/cycle_improvements/cycle_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/mixed_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/2way_fm_refinement/two_way_fm.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/two_way_flow_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/boundary_bfs.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/flow_solver.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/edge_cut_flow_solver.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/flow_solving_kernel/timer.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement/most_balanced_minimum_cuts/most_balanced_minimum_cuts.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/quotient_graph_scheduling.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/simple_quotient_graph_scheduler.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/quotient_graph_scheduling/active_block_quotient_graph_scheduler.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_core.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/kway_graph_refinement_commons.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/kway_graph_refinement/multitry_kway_fm.cpp
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/tabu_search/tabu_search.cpp
    # Galinier combine (part of core in modified_kahip)
    ${KAHIP_PARALLEL_DIR}/lib/parallel_mh/galinier_combine/gal_combine.cpp
    ${KAHIP_PARALLEL_DIR}/lib/parallel_mh/galinier_combine/construct_partition.cpp
    # Parallel MH sources (need MPI shim)
    ${KAHIP_PARALLEL_DIR}/lib/parallel_mh/parallel_mh_async.cpp
    ${KAHIP_PARALLEL_DIR}/lib/parallel_mh/population.cpp
    ${KAHIP_PARALLEL_DIR}/lib/parallel_mh/exchange/exchanger.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/graph_communication.cpp
    ${KAHIP_PARALLEL_DIR}/lib/tools/mpi_tools.cpp
)

# Fake mpi.h MUST come first so #include <mpi.h> finds our stub
target_include_directories(kahipe_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/shims/scc_mpi
    ${KAHIP_PARALLEL_DIR}/interface
    ${KAHIP_PARALLEL_DIR}/lib
    ${KAHIP_PARALLEL_DIR}/lib/io
    ${KAHIP_PARALLEL_DIR}/lib/partition
    ${KAHIP_PARALLEL_DIR}/lib/partition/uncoarsening/refinement/quotient_graph_refinement/flow_refinement
    ${KAHIP_PARALLEL_DIR}/lib/tools
    ${KAHIP_PARALLEL_DIR}
    ${KAHIP_PARALLEL_DIR}/app
    ${CMAKE_CURRENT_SOURCE_DIR}/external_repositories/KaHIP/extern/argtable3-3.2.2
)

target_compile_definitions(kahipe_static PRIVATE -DMODE_KAFFPA)
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

# --- Patch: MSVC doesn't support __attribute__((unused)) in sparsehash ---
file(READ ${HEISTREAM_DIR}/extern/sparsehash/src/sparsehash/internal/hashtable-common.h _SRC)
string(REPLACE
    "__attribute__((unused)) typedef SparsehashCompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]"
    "typedef SparsehashCompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]"
    _SRC "${_SRC}")
file(WRITE ${HEISTREAM_DIR}/extern/sparsehash/src/sparsehash/internal/hashtable-common.h "${_SRC}")

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

# --- _hypermis (HyperMIS reductions, always built) ---
pybind11_add_module(_hypermis bindings/hypermis_binding.cpp)
target_link_libraries(_hypermis PRIVATE hypermis_static)
install(TARGETS _hypermis DESTINATION chszlablib)

