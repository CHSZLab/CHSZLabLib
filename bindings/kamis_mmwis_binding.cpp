#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <memory>

#include "graph_access.h"
#include "graph_io.h"
#include "mmwis_config.h"
#include "mmwis_log.h"
#include "configuration_mis.h"
#include "reduction_evolution.h"
#include "branch_and_reduce_algorithm.h"

namespace py = pybind11;

static std::tuple<int, py::array_t<int>>
py_mmwis(py::array_t<int, py::array::c_style> xadj,
         py::array_t<int, py::array::c_style> adjncy,
         py::array_t<int, py::array::c_style> vwgt,
         double time_limit, int seed) {

    int n = static_cast<int>(xadj.size() - 1);
    int *xadj_ptr = xadj.mutable_data();
    int *adjncy_ptr = adjncy.mutable_data();

    graph_access G;
    if (vwgt.size() > 0) {
        std::vector<int> adjwgt(adjncy.size(), 1);
        G.build_from_metis_weighted(n, xadj_ptr, adjncy_ptr,
                                    vwgt.mutable_data(), adjwgt.data());
    } else {
        G.build_from_metis(n, xadj_ptr, adjncy_ptr);
    }

    mmwis::MISConfig mis_config;
    mmwis::configuration_mis cfg;
    cfg.mmwis(mis_config);
    mis_config.time_limit = time_limit;
    mis_config.evo_time_limit = time_limit / 10.0;
    mis_config.seed = seed;
    mis_config.console_log = false;
    mis_config.print_log = false;
    mis_config.write_solution = false;
    mis_config.write_kernel = false;
    mis_config.check_sorted = false;
    mis_config.weight_source = mmwis::MISConfig::Weight_Source::FILE;
    mis_config.perform_reductions = true;

    mmwis::mmwis_log::instance()->restart_overall_total_timer();
    mmwis::mmwis_log::instance()->set_config(mis_config);
    mmwis::mmwis_log::instance()->set_graph(G);

    // Suppress stdout
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    // Run evolutionary algorithm with reductions
    mmwis::mmwis_log::instance()->restart_total_timer();
    std::vector<bool> independent_set(G.number_of_nodes(), false);
    mmwis::reduction_evolution<mmwis::branch_and_reduce_algorithm> evo;
    std::vector<NodeID> best_nodes;
    std::vector<NodeID> worse_nodes;
    bool solved_exactly = false;
    evo.perform_mis_search(mis_config, G, independent_set, best_nodes,
                           worse_nodes, solved_exactly, false);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Collect IS vertices and weight
    std::vector<int> is_vertices;
    NodeWeight total_weight = 0;
    for (int i = 0; i < (int)G.number_of_nodes(); i++) {
        if (independent_set[i]) {
            is_vertices.push_back(i);
            total_weight += G.getNodeWeight(i);
        }
    }

    int is_weight = static_cast<int>(total_weight);
    py::array_t<int> result(is_vertices.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < is_vertices.size(); i++) {
        r(i) = is_vertices[i];
    }

    return std::make_tuple(is_weight, result);
}

PYBIND11_MODULE(_kamis_mmwis, m) {
    m.doc() = "Python bindings for KaMIS MMWIS (memetic weighted MIS)";

    m.def("mmwis_solver", &py_mmwis,
          py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
          py::arg("time_limit"), py::arg("seed"),
          R"doc(
          Run MMWIS memetic evolutionary weighted MIS algorithm.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (is_weight, is_vertices).
          )doc");
}
