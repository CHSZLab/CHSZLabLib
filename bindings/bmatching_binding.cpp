#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ds/modifiable_hypergraph.h"
#include "ds/bmatching.h"
#include "bmatching/greedy/ordered.h"
#include "bmatching/ils/ils.h"
#include "bmatching/reductions_sorted/driver.h"

namespace py = pybind11;

using HG = HeiHGM::BMatching::ds::StandardIntegerHypergraph;
using BM = HeiHGM::BMatching::ds::BMatching<HG>;

// Suppress stdout from TIMED_FUNC macros during algorithm execution.
struct StdoutSuppressor {
    std::streambuf *old_buf;
    std::ostringstream sink;
    StdoutSuppressor() : old_buf(std::cout.rdbuf(sink.rdbuf())) {}
    ~StdoutSuppressor() { std::cout.rdbuf(old_buf); }
};

// Weight-based evaluation function for greedy: higher weight = higher priority.
static double weight_eval(HG *hg, size_t e, BM &bm) {
    return static_cast<double>(hg->edgeWeight(e));
}

// Run the algorithm pipeline based on the algorithm string.
static void run_algorithm(HG &hg, BM &bm,
                          const std::string &algorithm,
                          int max_tries, double time_limit) {
    StdoutSuppressor suppressor;

    if (algorithm == "reduced" || algorithm == "reduced+ils") {
        HeiHGM::BMatching::bmatching::reductions_sorted::all_removals_exhaustive<BM>(bm, hg);
    }

    // Greedy (always run to get a baseline matching)
    HeiHGM::BMatching::bmatching::greedy_priority_ordered_matching<BM>(
        &hg, bm,
        [&](size_t e, BM &m) -> double {
            return static_cast<double>(hg.edgeWeight(e));
        });

    if (algorithm == "ils" || algorithm == "reduced+ils") {
        bm = HeiHGM::BMatching::bmatching::ils::iterated_local_search<BM>(
            bm, max_tries, time_limit);
    }
}

// Standard graph b-matching: receives CSR arrays.
static std::tuple<int, int, py::array_t<int64_t>>
py_bmatching(
        py::array_t<int64_t, py::array::c_style> xadj,
        py::array_t<int32_t, py::array::c_style> adjncy,
        py::array_t<int64_t, py::array::c_style> adjwgt,
        py::array_t<int32_t, py::array::c_style> capacities,
        const std::string &algorithm,
        double time_limit,
        int seed,
        int max_tries) {

    int n = static_cast<int>(xadj.size() - 1);
    const int64_t *xadj_ptr = xadj.data();
    const int32_t *adjncy_ptr = adjncy.data();
    const int64_t *adjwgt_ptr = adjwgt.data();
    const int32_t *cap_ptr = capacities.data();

    // Count unique edges (u < v)
    size_t num_edges = 0;
    for (int u = 0; u < n; u++) {
        for (int64_t idx = xadj_ptr[u]; idx < xadj_ptr[u + 1]; idx++) {
            if (u < adjncy_ptr[idx]) num_edges++;
        }
    }

    if (n == 0 || num_edges == 0) {
        py::array_t<int64_t> matched(0);
        return std::make_tuple(0, 0, matched);
    }

    // Set random seed
    HeiHGM::BMatching::utils::Randomize::instance().setSeed(seed);

    // Build hypergraph: each undirected edge becomes a 2-pin hyperedge
    HG hg(num_edges, static_cast<size_t>(n), true, true);

    // Map: edge_index -> hyperedge ID (in order of u < v traversal)
    for (int u = 0; u < n; u++) {
        for (int64_t idx = xadj_ptr[u]; idx < xadj_ptr[u + 1]; idx++) {
            int v = adjncy_ptr[idx];
            if (u < v) {
                int w = static_cast<int>(adjwgt_ptr[idx]);
                hg.addEdge({static_cast<size_t>(u), static_cast<size_t>(v)}, w);
            }
        }
    }

    // Set node capacities
    for (int i = 0; i < n; i++) {
        hg.setNodeWeight(static_cast<size_t>(i), static_cast<int>(cap_ptr[i]));
    }

    hg.sort_weight();

    BM bm(&hg);

    // Validate algorithm
    if (algorithm != "greedy" && algorithm != "ils" &&
        algorithm != "reduced" && algorithm != "reduced+ils") {
        throw py::value_error(
            "Unknown algorithm '" + algorithm + "'. "
            "Choose from: 'greedy', 'ils', 'reduced', 'reduced+ils'.");
    }

    run_algorithm(hg, bm, algorithm, max_tries, time_limit);

    // Extract matched edge indices
    int total_weight = bm.weight();
    int total_size = static_cast<int>(bm.size());

    py::array_t<int64_t> matched_edges(total_size);
    auto me = matched_edges.mutable_unchecked<1>();
    int idx = 0;
    for (auto e : bm.solution()) {
        me(idx++) = static_cast<int64_t>(e);
    }

    return std::make_tuple(total_weight, total_size, matched_edges);
}

// Hypergraph b-matching.
static std::tuple<int, int, py::array_t<int64_t>>
py_hypergraph_bmatching(
        int num_nodes,
        py::array_t<int64_t, py::array::c_style> edge_offsets,
        py::array_t<int32_t, py::array::c_style> edge_pins,
        py::array_t<int32_t, py::array::c_style> edge_weights_arr,
        py::array_t<int32_t, py::array::c_style> capacities,
        const std::string &algorithm,
        double time_limit,
        int seed,
        int max_tries) {

    const int64_t *off_ptr = edge_offsets.data();
    const int32_t *pin_ptr = edge_pins.data();
    const int32_t *ew_ptr = edge_weights_arr.data();
    const int32_t *cap_ptr = capacities.data();
    int num_edges = static_cast<int>(edge_offsets.size() - 1);

    if (num_nodes == 0 || num_edges == 0) {
        py::array_t<int64_t> matched(0);
        return std::make_tuple(0, 0, matched);
    }

    HeiHGM::BMatching::utils::Randomize::instance().setSeed(seed);

    HG hg(static_cast<size_t>(num_edges), static_cast<size_t>(num_nodes), true, true);

    for (int e = 0; e < num_edges; e++) {
        std::vector<size_t> pins;
        for (int64_t j = off_ptr[e]; j < off_ptr[e + 1]; j++) {
            pins.push_back(static_cast<size_t>(pin_ptr[j]));
        }
        hg.addEdge(pins, static_cast<int>(ew_ptr[e]));
    }

    for (int i = 0; i < num_nodes; i++) {
        hg.setNodeWeight(static_cast<size_t>(i), static_cast<int>(cap_ptr[i]));
    }

    hg.sort_weight();

    BM bm(&hg);

    if (algorithm != "greedy" && algorithm != "ils" &&
        algorithm != "reduced" && algorithm != "reduced+ils") {
        throw py::value_error(
            "Unknown algorithm '" + algorithm + "'. "
            "Choose from: 'greedy', 'ils', 'reduced', 'reduced+ils'.");
    }

    run_algorithm(hg, bm, algorithm, max_tries, time_limit);

    int total_weight = bm.weight();
    int total_size = static_cast<int>(bm.size());

    py::array_t<int64_t> matched_edges(total_size);
    auto me = matched_edges.mutable_unchecked<1>();
    int idx = 0;
    for (auto e : bm.solution()) {
        me(idx++) = static_cast<int64_t>(e);
    }

    return std::make_tuple(total_weight, total_size, matched_edges);
}

PYBIND11_MODULE(_bmatching, m) {
    m.doc() = "Python bindings for HeiHGM b-matching algorithms";

    m.def("bmatching", &py_bmatching,
          py::arg("xadj"), py::arg("adjncy"), py::arg("adjwgt"),
          py::arg("capacities"), py::arg("algorithm"),
          py::arg("time_limit"), py::arg("seed"), py::arg("max_tries"),
          R"doc(
          Compute a b-matching on a standard graph.

          Returns (weight, size, matched_edge_indices).
          )doc");

    m.def("hypergraph_bmatching", &py_hypergraph_bmatching,
          py::arg("num_nodes"), py::arg("edge_offsets"),
          py::arg("edge_pins"), py::arg("edge_weights"),
          py::arg("capacities"), py::arg("algorithm"),
          py::arg("time_limit"), py::arg("seed"), py::arg("max_tries"),
          R"doc(
          Compute a b-matching on a hypergraph.

          Returns (weight, size, matched_edge_indices).
          )doc");
}
