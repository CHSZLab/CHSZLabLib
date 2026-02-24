#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>

#include "graph_access.h"
#include "graph_io.h"
#include "mis_config.h"
#include "mis_log.h"
#include "configuration_mis.h"
#include "branch_and_reduce_algorithm.h"

namespace py = pybind11;

static std::tuple<int, py::array_t<int>>
py_branch_reduce(py::array_t<int, py::array::c_style> xadj,
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

    MISConfig mis_config;
    configuration_mis cfg;
    cfg.standard(mis_config);
    mis_config.time_limit = time_limit;
    mis_config.seed = seed;
    mis_config.console_log = false;
    mis_config.print_log = false;
    mis_config.write_graph = false;
    mis_config.check_sorted = false;
    mis_config.weight_source = MISConfig::Weight_Source::FILE;  // use weights from graph

    mis_log::instance()->restart_total_timer();
    mis_log::instance()->set_config(mis_config);
    mis_log::instance()->set_graph(G);

    // Suppress stdout
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    branch_and_reduce_algorithm reducer(G, mis_config);
    reducer.run_branch_reduce();
    NodeWeight mwis_weight = reducer.get_is_weight();
    reducer.apply_branch_reduce_solution(G);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Collect IS vertices
    std::vector<int> is_vertices;
    forall_nodes(G, node) {
        if (G.getPartitionIndex(node) == 1) {
            is_vertices.push_back(node);
        }
    } endfor

    int is_weight = static_cast<int>(mwis_weight);
    py::array_t<int> result(is_vertices.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < is_vertices.size(); i++) {
        r(i) = is_vertices[i];
    }

    return std::make_tuple(is_weight, result);
}

PYBIND11_MODULE(_kamis_wmis, m) {
    m.doc() = "Python bindings for KaMIS weighted MIS (branch & reduce)";

    m.def("branch_reduce", &py_branch_reduce,
          py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
          py::arg("time_limit"), py::arg("seed"),
          R"doc(
          Run weighted branch-and-reduce exact MIS algorithm.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (is_weight, is_vertices).
          )doc");
}
