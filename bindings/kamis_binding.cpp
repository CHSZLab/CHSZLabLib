#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>

#include "data_structure/graph_access.h"
#include "mis_config.h"
#include "mis_log.h"
#include "graph_io.h"
#include "reduction_evolution.h"
#include "mis/kernel/branch_and_reduce_algorithm.h"
#include "mis/kernel/ParFastKer/fast_reductions/src/full_reductions.h"
#include "ils/online_ils.h"

namespace py = pybind11;

// Set up a standard MISConfig without needing configuration_mis.h (avoids OMIS conflicts)
static void init_standard_config(MISConfig &c) {
    c.population_size = 50;
    c.repetitions = 50;
    c.time_limit = 1000.0;
    c.kahip_mode = 0; // FAST
    c.seed = 0;
    c.diversify = true;
    c.imbalance = 0.03;
    c.randomize_imbalance = true;
    c.enable_tournament_selection = true;
    c.tournament_size = 2;
    c.flip_coin = 1;
    c.use_hopcroft = false;
    c.optimize_candidates = true;
    c.use_multiway_vc = false;
    c.multiway_blocks = 64;
    c.insert_threshold = 150;
    c.pool_threshold = 250;
    c.pool_renewal_factor = 10.0;
    c.number_of_separators = 10;
    c.number_of_partitions = 10;
    c.number_of_k_separators = 10;
    c.number_of_k_partitions = 10;
    c.print_repetition = false;
    c.print_population = false;
    c.console_log = false;
    c.print_log = false;
    c.write_graph = false;
    c.check_sorted = false;
    c.ils_iterations = 15000;
    c.force_k = 1;
    c.force_cand = 4;
    c.all_reductions = true;
    c.reduction_threshold = 350;
    c.remove_fraction = 0.10;
    c.extract_best_nodes = true;
    c.start_greedy_adaptive = false;
    c.best_limit = 0;
    c.fullKernelization = false;
}

// Build graph_access from numpy CSR arrays
static void build_graph(graph_access &G,
                        py::array_t<int, py::array::c_style> &xadj,
                        py::array_t<int, py::array::c_style> &adjncy,
                        py::array_t<int, py::array::c_style> &vwgt) {
    int n = static_cast<int>(xadj.size() - 1);
    int *xadj_ptr = xadj.mutable_data();
    int *adjncy_ptr = adjncy.mutable_data();

    if (vwgt.size() > 0) {
        std::vector<int> adjwgt(adjncy.size(), 1);
        G.build_from_metis_weighted(n, xadj_ptr, adjncy_ptr,
                                    vwgt.mutable_data(), adjwgt.data());
    } else {
        G.build_from_metis(n, xadj_ptr, adjncy_ptr);
    }
}

// ---------------------------------------------------------------------------
// ReduMIS — evolutionary unweighted MIS
// ---------------------------------------------------------------------------
static std::tuple<int, py::array_t<int>>
py_redumis(py::array_t<int, py::array::c_style> xadj,
           py::array_t<int, py::array::c_style> adjncy,
           py::array_t<int, py::array::c_style> vwgt,
           double time_limit, int seed, bool full_kernelization) {

    graph_access G;
    build_graph(G, xadj, adjncy, vwgt);

    MISConfig mis_config;
    init_standard_config(mis_config);
    mis_config.time_limit = time_limit;
    mis_config.seed = seed;
    mis_config.fullKernelization = full_kernelization;

    mis_log::instance()->restart_total_timer();
    mis_log::instance()->set_config(mis_config);
    mis_log::instance()->set_graph(G);

    // Suppress stdout during algorithm run
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    std::vector<bool> independent_set(G.number_of_nodes(), false);
    std::vector<NodeID> best_nodes;

    if (full_kernelization) {
        reduction_evolution<branch_and_reduce_algorithm> evo;
        evo.perform_mis_search(mis_config, G, independent_set, best_nodes);
    } else {
        reduction_evolution<full_reductions> evo;
        evo.perform_mis_search(mis_config, G, independent_set, best_nodes);
    }

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Collect results
    std::vector<int> is_vertices;
    for (int i = 0; i < (int)G.number_of_nodes(); i++) {
        if (independent_set[i]) {
            is_vertices.push_back(i);
        }
    }

    int is_size = static_cast<int>(is_vertices.size());
    py::array_t<int> result(is_size);
    auto r = result.mutable_unchecked<1>();
    for (int i = 0; i < is_size; i++) {
        r(i) = is_vertices[i];
    }

    return std::make_tuple(is_size, result);
}

// ---------------------------------------------------------------------------
// OnlineMIS — online local search unweighted MIS
// ---------------------------------------------------------------------------
static std::tuple<int, py::array_t<int>>
py_online_mis(py::array_t<int, py::array::c_style> xadj,
              py::array_t<int, py::array::c_style> adjncy,
              py::array_t<int, py::array::c_style> vwgt,
              double time_limit, int seed, int ils_iterations) {

    graph_access G;
    build_graph(G, xadj, adjncy, vwgt);

    MISConfig mis_config;
    init_standard_config(mis_config);
    mis_config.time_limit = time_limit;
    mis_config.seed = seed;
    mis_config.ils_iterations = ils_iterations;

    mis_log::instance()->restart_total_timer();
    mis_log::instance()->set_config(mis_config);
    mis_log::instance()->set_graph(G);

    // Suppress stdout
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    online_ils online;
    online.perform_ils(mis_config, G, mis_config.ils_iterations);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Collect results: partition index == 1 means in IS
    std::vector<int> is_vertices;
    forall_nodes(G, node) {
        if (G.getPartitionIndex(node) == 1) {
            is_vertices.push_back(node);
        }
    } endfor

    int is_size = static_cast<int>(is_vertices.size());
    py::array_t<int> result(is_size);
    auto r = result.mutable_unchecked<1>();
    for (int i = 0; i < is_size; i++) {
        r(i) = is_vertices[i];
    }

    return std::make_tuple(is_size, result);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_kamis, m) {
    m.doc() = "Python bindings for KaMIS unweighted MIS (ReduMIS + OnlineMIS)";

    m.def("redumis", &py_redumis,
          py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
          py::arg("time_limit"), py::arg("seed"),
          py::arg("full_kernelization"),
          R"doc(
          Run ReduMIS evolutionary maximum independent set algorithm.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (is_size, is_vertices).
          )doc");

    m.def("online_mis", &py_online_mis,
          py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
          py::arg("time_limit"), py::arg("seed"),
          py::arg("ils_iterations"),
          R"doc(
          Run OnlineMIS local search algorithm.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (is_size, is_vertices).
          )doc");
}
