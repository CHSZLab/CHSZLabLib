#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Bmatching headers (header-only, templates)
#include "ds/modifiable_hypergraph.h"
#include "ds/bmatching.h"
#include "bmatching/greedy/ordered.h"
#include "bmatching/ils/ils.h"
#include "bmatching/reductions_sorted/driver.h"
#include "bmatching/reductions_sorted/foldings.h"
// Prevent computeIlp from calling exit() on Gurobi exceptions —
// rethrow as a C++ exception so Python can handle it gracefully.
#define exit(code) throw std::runtime_error("ILP solver failed (Gurobi error)")
#include "bmatching/ilp/ilp_exact.h"
#undef exit

namespace py = pybind11;

using Graph = HeiHGM::BMatching::ds::ModifiableHypergraph<>;
using BMatch = HeiHGM::BMatching::ds::BMatching<Graph>;

// Build a ModifiableHypergraph from CSR arrays + capacities
static std::unique_ptr<Graph> build_hypergraph(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<double, py::array::c_style> edge_weights,
    int num_nodes)
{
    auto ep = eptr.unchecked<1>();
    auto ev = everts.unchecked<1>();
    auto ew = edge_weights.unchecked<1>();
    int num_edges = static_cast<int>(eptr.size() - 1);

    // Match CLI: nodesInEdgesSorted=true, edgesInNodesSorted=true
    auto g = std::make_unique<Graph>(num_edges, num_nodes, true, true);

    for (int e = 0; e < num_edges; e++) {
        int start = static_cast<int>(ep(e));
        int end = static_cast<int>(ep(e + 1));
        std::vector<size_t> pins;
        pins.reserve(end - start);
        for (int i = start; i < end; i++) {
            pins.push_back(static_cast<size_t>(ev(i)));
        }
        // Sort pins to match CLI's sorted insertion
        std::sort(pins.begin(), pins.end());
        g->addEdge(pins, static_cast<int>(ew(e)));
    }
    return g;
}

// Run a static b-matching algorithm and return (matched_edge_indices, total_weight, is_optimal)
static std::tuple<py::array_t<int32_t>, double, bool>
py_bmatching(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<double, py::array::c_style> edge_weights,
    py::array_t<int32_t, py::array::c_style> capacities,
    int num_nodes,
    const std::string& algorithm,
    int seed,
    int ils_iterations,
    double ils_time_limit,
    double ILP_time_limit)
{
    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    // seed >= 0: explicit seed; seed < 0: reset to std::mt19937 default (5489)
    // to match CLI behavior where seed is never set (fresh process)
    HeiHGM::BMatching::utils::Randomize::instance().setSeed(seed >= 0 ? seed : 5489);

    auto graph = build_hypergraph(eptr, everts, edge_weights, num_nodes);

    // Set node capacities (node weights = capacities in b-matching)
    auto cap = capacities.unchecked<1>();
    for (int v = 0; v < num_nodes; v++) {
        graph->setNodeWeight(static_cast<size_t>(v), static_cast<int>(cap(v)));
    }

    BMatch bm(graph.get());
    bool is_optimal = false;

    using EdgeID = typename BMatch::EdgeID_t;

    if (algorithm == "greedy_random") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                return HeiHGM::BMatching::utils::Randomize::instance().getRandomFloat(0.0f, 1.0f);
            });
    } else if (algorithm == "greedy_weight_desc") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                return static_cast<double>(graph->edgeWeight(e));
            });
    } else if (algorithm == "greedy_weight_asc") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                return -static_cast<double>(graph->edgeWeight(e));
            });
    } else if (algorithm == "greedy_degree_asc") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                double deg_product = 1.0;
                for (auto p : graph->pins(e)) {
                    deg_product *= graph->nodeDegree(p);
                }
                return -deg_product;
            });
    } else if (algorithm == "greedy_degree_desc") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                double deg_product = 1.0;
                for (auto p : graph->pins(e)) {
                    deg_product *= graph->nodeDegree(p);
                }
                return deg_product;
            });
    } else if (algorithm == "greedy_weight_degree_ratio_desc") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                double cap_product = 1.0;
                for (auto p : graph->pins(e)) {
                    cap_product *= bm.capacity(p);
                }
                return static_cast<double>(graph->edgeWeight(e)) / cap_product;
            });
    } else if (algorithm == "greedy_weight_degree_ratio_asc") {
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                double cap_product = 1.0;
                for (auto p : graph->pins(e)) {
                    cap_product *= bm.capacity(p);
                }
                return -static_cast<double>(graph->edgeWeight(e)) / cap_product;
            });
    } else if (algorithm == "reductions") {
        // Reductions → ILP on reduced instance → unfold
        HeiHGM::BMatching::bmatching::reductions_sorted::all_removals_exhaustive(bm, *graph);
        // Check if the ILP has edges to solve (edges() skips disabled/empty)
        bool has_remaining = graph->edges().begin() != graph->edges().end();
        if (has_remaining) {
            // Restore stdout/stderr for Gurobi (gurobipy needs Python I/O)
            std::cout.rdbuf(old_cout);
            std::cerr.rdbuf(old_cerr);
            HeiHGM::BMatching::bmatching::ilp::computeIlp<BMatch>(
                graph.get(), bm, is_optimal, ILP_time_limit);
            // Re-suppress stdout/stderr
            std::cout.rdbuf(null_stream.rdbuf());
            std::cerr.rdbuf(null_stream.rdbuf());
        } else {
            is_optimal = true;
        }
        HeiHGM::BMatching::bmatching::reductions_sorted::weighted_vertex_unfolding(*graph, bm);
    } else if (algorithm == "ils") {
        // Greedy init + ILS — matches CLI chain: greedy(bweight) → ils
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                return static_cast<double>(graph->edgeWeight(e));
            });
        bm = HeiHGM::BMatching::bmatching::ils::iterated_local_search(
            bm, ils_iterations, ils_time_limit);
    } else {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
        throw std::invalid_argument("Unknown algorithm: " + algorithm);
    }

    // Collect matched edges
    std::vector<int32_t> matched;
    for (auto e : bm.solution()) {
        matched.push_back(static_cast<int32_t>(e));
    }
    double total_weight = static_cast<double>(bm.weight());

    // Restore stdout/stderr
    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Build numpy array
    py::array_t<int32_t> result(matched.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < matched.size(); i++) {
        r(i) = matched[i];
    }

    return std::make_tuple(result, total_weight, is_optimal);
}

PYBIND11_MODULE(_bmatching, m) {
    m.doc() = "Python bindings for HeiHGM b-matching on hypergraphs";

    m.def("bmatching", &py_bmatching,
          py::arg("eptr"), py::arg("everts"),
          py::arg("edge_weights"), py::arg("capacities"),
          py::arg("num_nodes"),
          py::arg("algorithm"), py::arg("seed"),
          py::arg("ils_iterations"), py::arg("ils_time_limit"),
          py::arg("ILP_time_limit"),
          R"doc(
          Run a hypergraph b-matching algorithm.

          Parameters
          ----------
          eptr : ndarray[int64]
              Edge pointer array (length num_edges + 1).
          everts : ndarray[int32]
              Concatenated vertex lists per edge.
          edge_weights : ndarray[float64]
              Edge weight array (length num_edges).
          capacities : ndarray[int32]
              Node capacity array (length num_nodes).
          num_nodes : int
              Number of vertices.
          algorithm : str
              Algorithm name.
          seed : int
              Random seed.
          ils_iterations : int
              Max ILS iterations (only for "ils").
          ils_time_limit : float
              ILS time limit in seconds (only for "ils").
          ILP_time_limit : float
              ILP time limit in seconds for "reductions" algorithm.

          Returns
          -------
          tuple[ndarray[int32], float, bool]
              (matched_edge_indices, total_weight, is_optimal).
          )doc");
}
