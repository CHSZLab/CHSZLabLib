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
    t = t.replace('#include "absl/container/flat_hash_map.h"',
                   '// #include "absl/container/flat_hash_map.h"  // patched out')
    t = t.replace('absl::flat_hash_map', 'std::unordered_map')
    # Fix: bare 'return;' in non-void set_decision (clang error)
    t = re.sub(r'(set_decision\(PartitionID block,.*?my_thread, degree\);)\s*return;',
               r'\1 return block;', t, flags=re.DOTALL)
    # Fix: const correctness in get_block_address (declaration + definition)
    t = t.replace('floating_block *get_block_address', 'const floating_block *get_block_address')
    t = re.sub(r'inline floating_block \*\s*\n(\s*vertex_partitioning::get_block_address)',
               r'inline const floating_block *\n\1', t)
    t = t.replace('  floating_block *block_pointer = &(blocks[block_id]);',
                   '  const floating_block *block_pointer = &(blocks[block_id]);')
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
    # comparison operators: add const qualifier
    t = re.sub(r'bool operator([<>=!]+)\(const minimal_min_heap<U>& obj\) \{',
               r'bool operator\1(const minimal_min_heap<U>& obj) const {', t)
    # get_min() needs a const overload for const comparison operators
    t = re.sub(r'(U& get_min\(\) \{[^}]+\})',
               r'\1 const U& get_min() const { return (*this->fp)[0]; }', t)
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

