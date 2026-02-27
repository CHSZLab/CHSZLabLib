#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>

#include "kaHIP_interface.h"
#include "balance_configuration.h"
#include "configuration.h"
#include "partition/partition_config.h"
#include "data_structure/graph_access.h"
#include "parallel_mh/parallel_mh_async.h"
#include "tools/quality_metrics.h"
#include "tools/random_functions.h"
#include "tools/macros_assertions.h"

namespace py = pybind11;

// ULTRAFASTSOCIAL is not in the original KaHIP interface header
#ifndef ULTRAFASTSOCIAL
#define ULTRAFASTSOCIAL 6
#endif

// ---------------------------------------------------------------------------
// kaffpaE wrapper — matches CLI (kaffpaE.cpp) exactly
// Uses same KaHIP/lib as the CLI binary.
// ---------------------------------------------------------------------------
static std::tuple<int, double, py::array_t<int>>
py_kaffpaE(py::array_t<int, py::array::c_style> vwgt,
           py::array_t<int, py::array::c_style> xadj,
           py::array_t<int, py::array::c_style> adjcwgt,
           py::array_t<int, py::array::c_style> adjncy,
           int nparts, double imbalance, bool suppress_output,
           bool graph_partitioned,
           py::array_t<int, py::array::c_style> initial_partition,
           int time_limit, int seed, int mode)
{
    int n = static_cast<int>(xadj.size() - 1);

    int* vwgt_ptr    = vwgt.size()    > 0 ? vwgt.mutable_data()    : nullptr;
    int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

    py::array_t<int> part(n);
    int edgecut  = 0;
    double balance = 0.0;

    // If warm-starting, copy the initial partition into the output array
    if (graph_partitioned && initial_partition.size() == n) {
        std::memcpy(part.mutable_data(), initial_partition.data(),
                    n * sizeof(int));
    }

    // --- Build config exactly as CLI kaffpaE.cpp + parse_parameters.h ---
    // CLI chain for MODE_KAFFPAE: standard() -> strong() -> user_mode()
    configuration cfg;
    PartitionConfig partition_config;
    partition_config.k = nparts;
    cfg.standard(partition_config);
    cfg.strong(partition_config);  // default for kaffpaE (parse_parameters.h line 230)

    switch (mode) {
        case FAST:           cfg.fast(partition_config); break;
        case ECO:            cfg.eco(partition_config); break;
        case STRONG:         cfg.strong(partition_config); break;
        case FASTSOCIAL:     cfg.fastsocial(partition_config); break;
        case ULTRAFASTSOCIAL:
            cfg.fastsocial(partition_config);
            // Original KaHIP doesn't have ultra_fast_kaffpaE_interfacecall;
            // approximate by using pool_size=1 so init creates only 1 individual
            partition_config.mh_pool_size = 1;
            break;
        case ECOSOCIAL:      cfg.ecosocial(partition_config); break;
        case STRONGSOCIAL:   cfg.strongsocial(partition_config); break;
        default:             cfg.eco(partition_config); break;
    }

    partition_config.seed       = seed;
    partition_config.k          = nparts;
    partition_config.imbalance  = 100.0 * imbalance;
    partition_config.time_limit = time_limit;

    // --- Build graph from CSR ---
    graph_access G;
    G.build_from_metis(n, xadj.mutable_data(), adjncy.mutable_data());
    G.set_partition_count(partition_config.k);

    srand(partition_config.seed);
    random_functions::setSeed(partition_config.seed);

    if (vwgt_ptr != nullptr) {
        forall_nodes(G, node) {
            G.setNodeWeight(node, vwgt_ptr[node]);
        } endfor
    }
    if (adjcwgt_ptr != nullptr) {
        forall_edges(G, e) {
            G.setEdgeWeight(e, adjcwgt_ptr[e]);
        } endfor
    }

    // --- Exactly matching CLI kaffpaE.cpp lines 62-68 ---
    partition_config.kaffpaE = true;
    if (partition_config.imbalance < 1) {
        partition_config.kabapE = true;
    }

    balance_configuration bc;
    bc.configurate_balance(partition_config, G);

    // --- Apply initial partition if provided ---
    if (graph_partitioned) {
        forall_nodes(G, node) {
            G.setPartitionIndex(node, part.mutable_data()[node]);
        } endfor
        partition_config.graph_allready_partitioned  = true;
        partition_config.no_new_initial_partitioning = true;
    }

    // --- Suppress stdout/stderr at fd level ---
    fflush(stdout);
    fflush(stderr);
    int old_stdout = dup(STDOUT_FILENO);
    int old_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    close(devnull);

    // --- Run evolutionary partitioner ---
    parallel_mh_async mh;
    mh.perform_partitioning(partition_config, G);

    fflush(stdout);
    fflush(stderr);
    dup2(old_stdout, STDOUT_FILENO);
    dup2(old_stderr, STDERR_FILENO);
    close(old_stdout);
    close(old_stderr);

    // --- Extract results ---
    forall_nodes(G, node) {
        part.mutable_data()[node] = G.getPartitionIndex(node);
    } endfor

    quality_metrics qm;
    edgecut = qm.edge_cut(G);
    balance = qm.balance(G);

    return std::make_tuple(edgecut, balance, part);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_kahipe, m) {
    m.doc() = "Python bindings for KaHIP evolutionary graph partitioning (kaffpaE)";

    // Mode constants
    m.attr("FAST")           = 0;
    m.attr("ECO")            = 1;
    m.attr("STRONG")         = 2;
    m.attr("FASTSOCIAL")     = 3;
    m.attr("ECOSOCIAL")      = 4;
    m.attr("STRONGSOCIAL")   = 5;
    m.attr("ULTRAFASTSOCIAL") = 6;

    m.def("kaffpaE", &py_kaffpaE,
          py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
          py::arg("adjncy"), py::arg("nparts"), py::arg("imbalance"),
          py::arg("suppress_output"), py::arg("graph_partitioned"),
          py::arg("initial_partition"),
          py::arg("time_limit"), py::arg("seed"), py::arg("mode"),
          R"doc(
          Run KaHIP's evolutionary/memetic graph partitioner (kaffpaE).

          Parameters
          ----------
          vwgt : ndarray[int32]
              Node weights (empty array for unit weights).
          xadj : ndarray[int32]
              CSR row pointers.
          adjcwgt : ndarray[int32]
              Edge weights (empty array for unit weights).
          adjncy : ndarray[int32]
              CSR column indices.
          nparts : int
              Number of partitions.
          imbalance : float
              Allowed imbalance (e.g. 0.03 = 3%).
          suppress_output : bool
              Suppress KaHIP console output.
          graph_partitioned : bool
              Whether an initial partition is provided.
          initial_partition : ndarray[int32]
              Initial partition (used when graph_partitioned is True).
          time_limit : int
              Time limit in seconds for the evolutionary algorithm.
          seed : int
              Random seed.
          mode : int
              Quality mode (FAST=0, ECO=1, STRONG=2, ...).

          Returns
          -------
          tuple[int, float, ndarray[int32]]
              (edgecut, balance, partition_assignment).
          )doc");
}
