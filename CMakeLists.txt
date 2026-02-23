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
