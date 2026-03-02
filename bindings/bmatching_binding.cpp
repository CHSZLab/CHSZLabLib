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

    auto g = std::make_unique<Graph>(num_edges, num_nodes, false, false);

    for (int e = 0; e < num_edges; e++) {
        int start = static_cast<int>(ep(e));
        int end = static_cast<int>(ep(e + 1));
        std::vector<size_t> pins;
        pins.reserve(end - start);
        for (int i = start; i < end; i++) {
            pins.push_back(static_cast<size_t>(ev(i)));
        }
        g->addEdge(pins, static_cast<int>(ew(e)));
    }
    g->sort();
    return g;
}

// Run a static b-matching algorithm and return (matched_edge_indices, total_weight)
static std::tuple<py::array_t<int32_t>, double>
py_bmatching(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<double, py::array::c_style> edge_weights,
    py::array_t<int32_t, py::array::c_style> capacities,
    int num_nodes,
    const std::string& algorithm,
    int seed,
    int ils_iterations,
    double ils_time_limit)
{
    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    HeiHGM::BMatching::utils::Randomize::instance().setSeed(seed);

    auto graph = build_hypergraph(eptr, everts, edge_weights, num_nodes);

    // Set node capacities (node weights = capacities in b-matching)
    auto cap = capacities.unchecked<1>();
    for (int v = 0; v < num_nodes; v++) {
        graph->setNodeWeight(static_cast<size_t>(v), static_cast<int>(cap(v)));
    }

    BMatch bm(graph.get());

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
        // Greedy initial solution, then reductions + unfold
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                return static_cast<double>(graph->edgeWeight(e));
            });
        HeiHGM::BMatching::bmatching::reductions_sorted::all_removals_exhaustive(bm, *graph);
        bm.maximize();
    } else if (algorithm == "ils") {
        // Greedy initial solution, then ILS
        HeiHGM::BMatching::bmatching::greedy_static_ordered_matching<BMatch>(
            graph.get(), bm,
            [&](EdgeID e) -> double {
                return static_cast<double>(graph->edgeWeight(e));
            });
        HeiHGM::BMatching::bmatching::ils::iterated_local_searc_inplace(
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

    return std::make_tuple(result, total_weight);
}

PYBIND11_MODULE(_bmatching, m) {
    m.doc() = "Python bindings for HeiHGM b-matching on hypergraphs";

    m.def("bmatching", &py_bmatching,
          py::arg("eptr"), py::arg("everts"),
          py::arg("edge_weights"), py::arg("capacities"),
          py::arg("num_nodes"),
          py::arg("algorithm"), py::arg("seed"),
          py::arg("ils_iterations"), py::arg("ils_time_limit"),
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

          Returns
          -------
          tuple[ndarray[int32], float]
              (matched_edge_indices, total_weight).
          )doc");
}
