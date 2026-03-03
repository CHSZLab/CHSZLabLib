#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "data_structure/dyn_graph_access.h"
#include "DynWMISConfig.h"
#include "definitions_wmis.h"
#include "algorithms/dyn_wmis_algorithm.h"
#include "tools/algorithm_factory.h"

namespace py = pybind11;

static DynWMISAlgorithmType parse_algorithm(const std::string& name) {
    if (name == "simple" || name == "one_fast")   return DYNWMIS_SIMPLE;
    if (name == "greedy")                         return DYNWMIS_GREEDY;
    if (name == "deg_greedy")                     return DYNWMIS_GREEDY_DEG;
    if (name == "bfs")                            return DYNWMIS_BFS;
    if (name == "static" || name == "one_strong") return DYNWMIS_STATIC;
    throw std::invalid_argument("Unknown algorithm: " + name);
}

class DynWMISSolver {
public:
    DynWMISSolver(int num_nodes,
                  py::array_t<int32_t, py::array::c_style> node_weights,
                  const std::string& algorithm,
                  int seed,
                  int bfs_depth,
                  double time_limit)
        : num_nodes_(num_nodes), algorithm_name_(algorithm)
    {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        G_ = std::make_shared<dyn_graph_access>(static_cast<NodeID>(num_nodes));

        // Set node weights
        auto w = node_weights.unchecked<1>();
        for (int i = 0; i < num_nodes; i++) {
            G_->setNodeWeight(static_cast<NodeID>(i),
                              static_cast<NodeWeight>(w(i)));
        }

        DynWMISConfig config;
        config.algorithmType = parse_algorithm(algorithm);
        config.seed = seed;
        config.bfs_depth = bfs_depth;
        config.local_solver_time_limit = time_limit;
        config.unit_weights = false;

        solver_ = getdyn_wmis_instance(config.algorithmType, G_, config);

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    void insert_edge(int u, int v) {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        solver_->handleInsertion(static_cast<NodeID>(u), static_cast<NodeID>(v));

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    void delete_edge(int u, int v) {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        solver_->handleDeletion(static_cast<NodeID>(u), static_cast<NodeID>(v));

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    int get_weight() {
        return static_cast<int>(solver_->weight());
    }

    py::array_t<bool> get_mis() {
        auto mis = solver_->getMIS();
        py::array_t<bool> result(num_nodes_);
        auto r = result.mutable_unchecked<1>();
        for (int i = 0; i < num_nodes_; i++) {
            r(i) = (i < static_cast<int>(mis.size())) ? mis[i] : false;
        }
        return result;
    }

private:
    int num_nodes_;
    std::string algorithm_name_;
    std::shared_ptr<dyn_graph_access> G_;
    std::unique_ptr<dyn_wmis_algorithm> solver_;
};

PYBIND11_MODULE(_dyn_wmis, m) {
    m.doc() = "Python bindings for DynWMIS (dynamic weighted MIS)";

    py::class_<DynWMISSolver>(m, "DynWMISSolver")
        .def(py::init<int, py::array_t<int32_t, py::array::c_style>,
                       const std::string&, int, int, double>(),
             py::arg("num_nodes"), py::arg("node_weights"),
             py::arg("algorithm"), py::arg("seed"),
             py::arg("bfs_depth"), py::arg("time_limit"),
             R"doc(
             Create a dynamic weighted MIS solver.

             Parameters
             ----------
             num_nodes : int
                 Number of vertices.
             node_weights : ndarray[int32]
                 Node weight array (length num_nodes).
             algorithm : str
                 Algorithm name.
             seed : int
                 Random seed.
             bfs_depth : int
                 BFS depth for local algorithms.
             time_limit : float
                 Time limit for local solver in seconds.
             )doc")
        .def("insert_edge", &DynWMISSolver::insert_edge,
             py::arg("u"), py::arg("v"),
             "Insert an undirected edge (u, v).")
        .def("delete_edge", &DynWMISSolver::delete_edge,
             py::arg("u"), py::arg("v"),
             "Delete an undirected edge (u, v).")
        .def("get_weight", &DynWMISSolver::get_weight,
             "Return current MIS total weight.")
        .def("get_mis", &DynWMISSolver::get_mis,
             "Return bool array: True if vertex is in MIS.");
}
