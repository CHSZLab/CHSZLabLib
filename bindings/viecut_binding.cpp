#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

#include "algorithms/global_mincut/algorithms.h"
#include "algorithms/global_mincut/minimum_cut.h"
#include "common/configuration.h"
#include "common/definitions.h"
#include "data_structure/graph_access.h"
#include "data_structure/mutable_graph.h"
#include "tools/random_functions.h"

#ifdef PARALLEL
#include <omp.h>
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// minimum_cut wrapper
// ---------------------------------------------------------------------------
static std::tuple<uint64_t, py::array_t<int32_t>>
py_minimum_cut(py::array_t<int32_t, py::array::c_style> xadj,
               py::array_t<int32_t, py::array::c_style> adjncy,
               py::array_t<int32_t, py::array::c_style> adjwgt,
               const std::string& algorithm,
               bool save_cut,
               int seed,
               int threads)
{
    int n = static_cast<int>(xadj.size() - 1);
    if (n <= 0) {
        throw std::invalid_argument("Graph must have at least one node");
    }

    const int32_t* xadj_ptr   = xadj.data();
    const int32_t* adjncy_ptr = adjncy.data();
    const int32_t* adjwgt_ptr = adjwgt.data();
    int total_edges = xadj_ptr[n];

    // Build graph_access manually from CSR arrays.
    // We avoid build_from_metis_weighted because graph_access::setNodeWeight
    // contains assert(false) and would crash in debug builds.
    auto ga = std::make_shared<graph_access>();
    ga->start_construction(static_cast<NodeID>(n),
                           static_cast<EdgeID>(total_edges));

    for (int i = 0; i < n; ++i) {
        ga->new_node();
        ga->setPartitionIndex(static_cast<NodeID>(i), 0);
        for (int e = xadj_ptr[i]; e < xadj_ptr[i + 1]; ++e) {
            EdgeID eid = ga->new_edge(static_cast<NodeID>(i),
                                      static_cast<NodeID>(adjncy_ptr[e]));
            ga->setEdgeWeight(eid, static_cast<EdgeWeight>(adjwgt_ptr[e]));
        }
    }
    ga->finish_construction();

    // Convert to mutable_graph (required by most algorithms)
    mutableGraphPtr G = mutable_graph::from_graph_access(ga);

    // Configure
    auto cfg = configuration::getConfig();
    cfg->save_cut = save_cut;
    cfg->seed = static_cast<size_t>(seed);
    random_functions::setSeed(seed);

#ifdef PARALLEL
    {
        int nthreads = threads > 0 ? threads : omp_get_max_threads();
        cfg->threads = static_cast<size_t>(nthreads);
        omp_set_num_threads(nthreads);
    }
#endif

    // Suppress stdout/stderr at fd level
    fflush(stdout);
    fflush(stderr);
    int old_stdout = dup(STDOUT_FILENO);
    int old_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    close(devnull);

    // Select and run algorithm
    minimum_cut* mc = selectMincutAlgorithm<mutableGraphPtr>(algorithm);
    EdgeWeight cut_value = mc->perform_minimum_cut(G);
    delete mc;

    fflush(stdout);
    fflush(stderr);
    dup2(old_stdout, STDOUT_FILENO);
    dup2(old_stderr, STDERR_FILENO);
    close(old_stdout);
    close(old_stderr);

    // Extract partition (per original node: 0 or 1)
    py::array_t<int32_t> partition(n);
    int32_t* part_ptr = partition.mutable_data();

    if (save_cut) {
        for (int i = 0; i < n; ++i) {
            part_ptr[i] = G->getNodeInCut(static_cast<NodeID>(i)) ? 1 : 0;
        }
    } else {
        // If save_cut was false, partition info is not available
        for (int i = 0; i < n; ++i) {
            part_ptr[i] = 0;
        }
    }

    return std::make_tuple(cut_value, partition);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_viecut, m) {
    m.doc() = "Python bindings for VieCut minimum cut algorithms";

    m.def("minimum_cut", &py_minimum_cut,
          py::arg("xadj"),
          py::arg("adjncy"),
          py::arg("adjwgt"),
          py::arg("algorithm") = "inexact",
          py::arg("save_cut") = true,
          py::arg("seed") = 0,
          py::arg("threads") = 0,
          R"doc(
          Compute a global minimum cut using VieCut.

          Parameters
          ----------
          xadj : ndarray[int32]
              CSR row pointers, shape (n+1,).
          adjncy : ndarray[int32]
              CSR column indices.
          adjwgt : ndarray[int32]
              Edge weights (same length as adjncy).
          algorithm : str
              Algorithm name: "inexact" (heuristic VieCut), "exact" (shared-memory exact), "cactus" (all minimum cuts).
          save_cut : bool
              If True, compute and return the partition assignment.
          seed : int
              Random seed.
          threads : int
              Number of threads (0 = all available cores).

          Returns
          -------
          tuple[int, ndarray[int32]]
              (cut_value, partition) where partition is a 0/1 array of length n.
          )doc");
}
