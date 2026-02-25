#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>

#include "definitions.h"
#include "datastructure/hypergraph.h"
#include "datastructure/fast_set.h"
#include "MIS_algorithm.h"
#include "config.h"

namespace py = pybind11;

// Forward declaration of hypergraph_init (defined in hypergraph.cpp but not in header)
extern hypergraph *hypergraph_init(NodeID n, NodeID m);
extern void hypergraph_append_element(NodeID *l, NodeID *a, NodeID **A, NodeID v);

// Build a hypergraph struct from edge-to-vertex CSR arrays
static hypergraph* build_hypergraph_from_csr(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes)
{
    auto eptr_buf = eptr.unchecked<1>();
    auto everts_buf = everts.unchecked<1>();
    int num_edges = static_cast<int>(eptr.size() - 1);

    hypergraph* g = hypergraph_init(
        static_cast<NodeID>(num_nodes),
        static_cast<NodeID>(num_edges));

    // Fill E (edge -> vertices) and V (vertex -> edges)
    for (int e = 0; e < num_edges; e++) {
        int start = static_cast<int>(eptr_buf(e));
        int end   = static_cast<int>(eptr_buf(e + 1));
        for (int i = start; i < end; i++) {
            NodeID v = static_cast<NodeID>(everts_buf(i));
            hypergraph_append_element(
                g->Ed + e, g->Ea + e, g->E + e, v);
            hypergraph_append_element(
                g->Vd + v, g->Va + v, g->V + v, static_cast<NodeID>(e));
        }
    }

    return g;
}

static std::tuple<int, py::array_t<int32_t>, double>
py_hypermis_reduce(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes,
    double time_limit,
    int seed,
    bool strong_reductions)
{
    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    // Configure globals
    VERBOSE = 0;
    TIME_KERNEL_SECONDS = static_cast<size_t>(time_limit);
    if (strong_reductions) {
        UNCONFINED_REDUCE = 1;
        NUM_REMOVED_EDGES = 1000000;
        CONSTANT_UNCONFINED = 5;
        ITERATIONS_UNCONFINED = 20000;
        EDGE_SIZE = 5000;
        HEURISTIC_RED = 1;
    } else {
        UNCONFINED_REDUCE = 0;
        REDUCE = 1;
        HEURISTIC_RED = 0;
        NUM_REMOVED_EDGES = 20000;
        ITERATIONS_UNCONFINED = 10000;
        CONSTANT_UNCONFINED = 3;
        EDGE_SIZE = 200;
    }

    // Build hypergraph from CSR
    hypergraph* g = build_hypergraph_from_csr(eptr, everts, num_nodes);

    // Create algorithm object and build neighbor lists
    MISH_algorithm* mis_alg = new MISH_algorithm(g);
    hypergraph_build_neighbors(g, &(mis_alg->node_set));

    // Run reductions
    auto start_time = std::chrono::high_resolution_clock::now();
    mis_alg->reduce_graph();
    auto end_time = std::chrono::high_resolution_clock::now();
    double reduction_time = std::chrono::duration<double>(
        end_time - start_time).count();

    // Collect IS vertices (those marked as included by reductions)
    std::vector<int32_t> is_vertices;
    for (NodeID v = 0; v < g->n; v++) {
        if (mis_alg->status.node_status[v] ==
            MISH_algorithm::IS_status::included) {
            is_vertices.push_back(static_cast<int32_t>(v));
        }
    }

    int offset = static_cast<int>(mis_alg->status.IS_size);

    // Clean up
    hypergraph_free(g);
    delete mis_alg;

    // Restore stdout/stderr
    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Build result numpy array
    py::array_t<int32_t> result(is_vertices.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < is_vertices.size(); i++) {
        r(i) = is_vertices[i];
    }

    return std::make_tuple(offset, result, reduction_time);
}

PYBIND11_MODULE(_hypermis, m) {
    m.doc() = "Python bindings for HyperMIS (hypergraph MIS reductions)";

    m.def("reduce", &py_hypermis_reduce,
          py::arg("eptr"), py::arg("everts"),
          py::arg("num_nodes"),
          py::arg("time_limit"), py::arg("seed"),
          py::arg("strong_reductions"),
          R"doc(
          Run HyperMIS reduction rules on a hypergraph.

          Parameters
          ----------
          eptr : ndarray[int64]
              Edge pointer array (length num_edges + 1).
          everts : ndarray[int32]
              Concatenated vertex lists per edge.
          num_nodes : int
              Number of vertices.
          time_limit : float
              Reduction time budget in seconds.
          seed : int
              Random seed.
          strong_reductions : bool
              Enable aggressive reduction rules.

          Returns
          -------
          tuple[int, ndarray[int32], float]
              (offset, is_vertices, reduction_time_seconds).
          )doc");
}
