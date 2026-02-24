cmake_minimum_required(VERSION 3.15)
project(chszlablib LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# Fix missing <cstdint> includes in older tlx/VieCut headers on newer compilers
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-include;cstdint>")

# =============================================================================
# KaHIP  (provides kahip_static target)
# =============================================================================
set(NOMPI ON CACHE BOOL "" FORCE)
set(NONATIVEOPTIMIZATIONS ON CACHE BOOL "" FORCE)
set(PARHIP OFF CACHE BOOL "" FORCE)
set(BUILDPYTHONMODULE OFF CACHE BOOL "" FORCE)
add_subdirectory(KaHIP EXCLUDE_FROM_ALL)

# =============================================================================
# VieClus  (built manually to avoid target-name conflicts with KaHIP)
# =============================================================================
# VieClus carries its own copy of KaHIP sources under extern/KaHIP.
# We compile everything into a single static library.

set(VIECLUS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VieClus)

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
set(VIECUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VieCut)

# tlx is the only compiled dependency VieCut needs
add_subdirectory(${VIECUT_DIR}/extlib/tlx EXCLUDE_FROM_ALL)

# =============================================================================
# CHILS  (plain C library -- no existing CMake)
# =============================================================================
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

# =============================================================================
# KaMIS  (Maximum Independent Set — three isolated static libraries)
# =============================================================================
# KaMIS has three sub-projects (root, wmis, mmwis) each with their own copy of
# KaHIP and conflicting symbols.  We build each as a separate static lib.

set(KAMIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/KaMIS)

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
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/fast_reductions/src/MaximumMatching.cpp
    ${KAMIS_DIR}/lib/mis/kernel/ParFastKer/fast_reductions/src/parallel_reductions.cpp
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
target_compile_options(kamis_redumis_static PRIVATE -w -fpermissive)

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

target_compile_options(kamis_wmis_static PRIVATE -w -fpermissive)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kamis_wmis_static PUBLIC OpenMP::OpenMP_CXX)
endif()

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

target_compile_options(kamis_mmwis_static PRIVATE -w -fpermissive)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kamis_mmwis_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# ScalableCorrelationClustering (SCC)
# =============================================================================
set(SCC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SCC)

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

target_compile_options(scc_static PRIVATE -w -fpermissive)

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

target_compile_options(scc_evo_static PRIVATE -w -fpermissive)

if(OpenMP_CXX_FOUND)
    target_link_libraries(scc_evo_static PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# Pybind11 modules
# =============================================================================

# --- _kahip ---
pybind11_add_module(_kahip bindings/kahip_binding.cpp)
target_link_libraries(_kahip PRIVATE kahip_static)
install(TARGETS _kahip DESTINATION chszlablib)

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
