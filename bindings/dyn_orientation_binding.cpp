#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "data_structure/dyn_graph_access.h"
#include "DeltaOrientationsConfig.h"
#include "DeltaOrientationsResult.h"
#include "algorithms/DynEdgeOrientation.h"
#include "tools/algorithm_factory.h"

namespace py = pybind11;

static DeltaOrientationsAlgorithmType parse_algorithm(const std::string& name) {
    if (name == "bfs")              return BFSCS;
    if (name == "naive_opt")        return NAIVEOPT;
    if (name == "impro_opt")        return IMPROOPT;
    if (name == "kflips")           return KFLIPSCS;
    if (name == "rwalk")            return RWALKCS;
    if (name == "naive")            return NAIVE;
    if (name == "brodal_fagerberg") return BRODAL_FAGERBERGCS;
    if (name == "max_descending")   return MAXDECENDING;
    if (name == "strong_opt")       return STRONG_OPT;
    if (name == "strong_opt_dfs")   return STRONG_OPT_DFS;
    if (name == "improved_opt")     return IMPROVED_OPT;
    if (name == "improved_opt_dfs") return IMPROVED_OPT_DFS;
    throw std::invalid_argument("Unknown algorithm: " + name);
}

class DynOrientationSolver {
public:
    DynOrientationSolver(int num_nodes, const std::string& algorithm, int seed)
        : num_nodes_(num_nodes), algorithm_name_(algorithm)
    {
        // Suppress stdout/stderr from C++ library
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        G_ = std::make_shared<dyn_graph_access>(static_cast<NodeID>(num_nodes));

        DeltaOrientationsConfig config;
        config.algorithmType = parse_algorithm(algorithm);
        config.seed = seed;

        result_ = std::make_unique<DeltaOrientationsResult>();
        solver_ = getdyn_edge_orientation_instance(config.algorithmType, G_, config, *result_);

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

    int get_max_out_degree() {
        return static_cast<int>(G_->maxDegree());
    }

    py::array_t<int32_t> get_out_degrees() {
        py::array_t<int32_t> degrees(num_nodes_);
        auto d = degrees.mutable_unchecked<1>();
        for (int i = 0; i < num_nodes_; i++) {
            d(i) = static_cast<int32_t>(G_->getNodeDegree(static_cast<NodeID>(i)));
        }
        return degrees;
    }

    int get_num_edges() {
        return static_cast<int>(G_->number_of_edges());
    }

private:
    int num_nodes_;
    std::string algorithm_name_;
    std::shared_ptr<dyn_graph_access> G_;
    std::unique_ptr<DeltaOrientationsResult> result_;
    std::shared_ptr<dyn_edge_orientation> solver_;
};

PYBIND11_MODULE(_dyn_orientation, m) {
    m.doc() = "Python bindings for DynDeltaOrientation (dynamic edge orientation)";

    py::class_<DynOrientationSolver>(m, "DynOrientationSolver")
        .def(py::init<int, const std::string&, int>(),
             py::arg("num_nodes"), py::arg("algorithm"), py::arg("seed"),
             R"doc(
             Create a dynamic edge orientation solver.

             Parameters
             ----------
             num_nodes : int
                 Number of vertices.
             algorithm : str
                 Algorithm name.
             seed : int
                 Random seed.
             )doc")
        .def("insert_edge", &DynOrientationSolver::insert_edge,
             py::arg("u"), py::arg("v"),
             "Insert an undirected edge (u, v).")
        .def("delete_edge", &DynOrientationSolver::delete_edge,
             py::arg("u"), py::arg("v"),
             "Delete an undirected edge (u, v).")
        .def("get_max_out_degree", &DynOrientationSolver::get_max_out_degree,
             "Return the current maximum out-degree.")
        .def("get_out_degrees", &DynOrientationSolver::get_out_degrees,
             "Return out-degree array for all nodes.")
        .def("get_num_edges", &DynOrientationSolver::get_num_edges,
             "Return current number of directed edges.");
}
