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

// Configure HyperMIS global variables.
// strong_reductions: enable unconfined vertex reductions + aggressive thresholds
// heuristic: enable heuristic peeling when reductions stall (greedy, not exact)
static void configure_hypermis(double time_limit, bool strong_reductions, bool heuristic) {
    VERBOSE = 0;
    TIME_KERNEL_SECONDS = static_cast<size_t>(time_limit);
    HEURISTIC_RED = heuristic ? 1 : 0;
    if (strong_reductions) {
        UNCONFINED_REDUCE = 1;
        NUM_REMOVED_EDGES = 1000000;
        CONSTANT_UNCONFINED = 5;
        ITERATIONS_UNCONFINED = 20000;
        EDGE_SIZE = 5000;
    } else {
        UNCONFINED_REDUCE = 0;
        REDUCE = 1;
        NUM_REMOVED_EDGES = 20000;
        ITERATIONS_UNCONFINED = 10000;
        CONSTANT_UNCONFINED = 3;
        EDGE_SIZE = 200;
    }
}

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

// Reduce and extract the kernel hypergraph as CSR arrays + remap.
// Returns: (offset, fixed_vertices, kernel_eptr, kernel_everts,
//           kernel_num_nodes, remap, reduction_time)
static std::tuple<int, py::array_t<int32_t>,
                  py::array_t<int64_t>, py::array_t<int32_t>,
                  int, py::array_t<int32_t>, double>
py_hypermis_reduce_and_extract_kernel(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes,
    double time_limit,
    int seed,
    bool strong_reductions,
    bool heuristic)
{
    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    configure_hypermis(time_limit, strong_reductions, heuristic);

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

    // Collect fixed IS vertices
    std::vector<int32_t> is_vertices;
    for (NodeID v = 0; v < g->n; v++) {
        if (mis_alg->status.node_status[v] ==
            MISH_algorithm::IS_status::included) {
            is_vertices.push_back(static_cast<int32_t>(v));
        }
    }
    int offset = static_cast<int>(mis_alg->status.IS_size);
    int remaining = static_cast<int>(mis_alg->status.remaining_nodes);

    // Extract kernel hypergraph
    std::vector<int64_t> k_eptr;
    std::vector<int32_t> k_everts;
    std::vector<int32_t> remap_vec;
    int kernel_num_nodes = 0;

    if (remaining > 0) {
        std::vector<NodeID> remap_raw;
        std::vector<bool> sol(g->n, false);
        hypergraph* rg = mis_alg->build_reduced_hypergraph(g, remap_raw, sol);

        kernel_num_nodes = static_cast<int>(rg->n);
        remap_vec.resize(rg->n);
        for (size_t i = 0; i < rg->n; i++)
            remap_vec[i] = static_cast<int32_t>(remap_raw[i]);

        // Build CSR from reduced hypergraph
        k_eptr.resize(rg->m + 1);
        k_eptr[0] = 0;
        for (NodeID e = 0; e < rg->m; e++) {
            k_eptr[e + 1] = k_eptr[e] + static_cast<int64_t>(rg->Ed[e]);
            for (NodeID j = 0; j < rg->Ed[e]; j++) {
                k_everts.push_back(static_cast<int32_t>(rg->E[e][j]));
            }
        }
        hypergraph_free(rg);
    }

    // Clean up
    hypergraph_free(g);
    delete mis_alg;

    // Restore stdout/stderr
    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Build numpy arrays
    py::array_t<int32_t> py_is_verts(is_vertices.size());
    auto r1 = py_is_verts.mutable_unchecked<1>();
    for (size_t i = 0; i < is_vertices.size(); i++)
        r1(i) = is_vertices[i];

    py::array_t<int64_t> py_keptr(k_eptr.size());
    auto r2 = py_keptr.mutable_unchecked<1>();
    for (size_t i = 0; i < k_eptr.size(); i++)
        r2(i) = k_eptr[i];

    py::array_t<int32_t> py_keverts(k_everts.size());
    auto r3 = py_keverts.mutable_unchecked<1>();
    for (size_t i = 0; i < k_everts.size(); i++)
        r3(i) = k_everts[i];

    py::array_t<int32_t> py_remap(remap_vec.size());
    auto r4 = py_remap.mutable_unchecked<1>();
    for (size_t i = 0; i < remap_vec.size(); i++)
        r4(i) = remap_vec[i];

    return std::make_tuple(offset, py_is_verts, py_keptr, py_keverts,
                           kernel_num_nodes, py_remap, reduction_time);
}

static std::tuple<int, py::array_t<int32_t>, double>
py_hypermis_reduce(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes,
    double time_limit,
    int seed,
    bool strong_reductions,
    bool heuristic)
{
    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    configure_hypermis(time_limit, strong_reductions, heuristic);

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
          py::arg("heuristic"),
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
              Enable aggressive reduction rules (unconfined vertices, larger thresholds).
          heuristic : bool
              Enable heuristic peeling when reductions stall (greedy, not exact).

          Returns
          -------
          tuple[int, ndarray[int32], float]
              (offset, is_vertices, reduction_time_seconds).
          )doc");

    m.def("reduce_and_extract_kernel", &py_hypermis_reduce_and_extract_kernel,
          py::arg("eptr"), py::arg("everts"),
          py::arg("num_nodes"),
          py::arg("time_limit"), py::arg("seed"),
          py::arg("strong_reductions"),
          py::arg("heuristic"),
          R"doc(
          Run HyperMIS reductions and extract the remaining kernel.

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
              Enable aggressive reduction rules (unconfined vertices, larger thresholds).
          heuristic : bool
              Enable heuristic peeling when reductions stall (greedy, not exact).

          Returns
          -------
          tuple[int, ndarray[int32], ndarray[int64], ndarray[int32], int, ndarray[int32], float]
              (offset, fixed_vertices, kernel_eptr, kernel_everts,
               kernel_num_nodes, remap, reduction_time_seconds).
              remap maps kernel vertex IDs back to original vertex IDs.
              If kernel is empty (reductions solved everything),
              kernel_eptr/kernel_everts are empty and kernel_num_nodes is 0.
          )doc");
}
