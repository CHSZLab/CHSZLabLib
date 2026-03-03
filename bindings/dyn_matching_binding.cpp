#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <climits>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "data_structure/dyn_graph_access.h"
#include "match_config.h"
#include "definitions.h"
#include "tools/random_functions.h"
#include "algorithms/dyn_matching.h"
#include "algorithms/rw_dyn_matching.h"
#include "algorithms/baswanaguptasen_dyn_matching.h"
#include "algorithms/neimansolomon_dyn_matching.h"
#include "algorithms/naive_dyn_matching.h"
#include "algorithms/blossom_dyn_matching.h"
#include "algorithms/blossom_dyn_matching_naive.h"
#include "algorithms/static_blossom.h"

namespace py = pybind11;

static AlgorithmType parse_algorithm(const std::string& name) {
    if (name == "random_walk")        return RANDOM_WALK;
    if (name == "baswana_gupta_sen")  return BASWANA_GUPTA_SENG;
    if (name == "neiman_solomon")     return NEIMAN_SOLOMON;
    if (name == "naive")              return NAIVE;
    if (name == "blossom")            return DYNBLOSSOM;
    if (name == "blossom_naive")      return DYNBLOSSOMNAIVE;
    if (name == "static_blossom")     return BLOSSOM;
    throw std::invalid_argument("Unknown algorithm: " + name);
}

class DynMatchingSolver {
public:
    DynMatchingSolver(int num_nodes, const std::string& algorithm, int seed)
        : num_nodes_(num_nodes), algorithm_name_(algorithm)
    {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        // Match CLI PRNG initialization
        srand(seed);
        random_functions::setSeed(seed);

        G_ = std::make_unique<dyn_graph_access>(static_cast<NodeID>(num_nodes));

        MatchConfig config;
        config.algorithm = parse_algorithm(algorithm);
        config.seed = seed;
        // Match CLI defaults (configuration.h standard())
        config.post_blossom = false;
        config.fast_rw = false;
        config.dynblossom_speedheuristic = false;
        config.dynblossom_weakspeedheuristic = false;
        config.blossom_init = BLOSSOMEXTRAGREEDY;
        config.rw_max_length = 10;
        config.rw_low_degree_settle = false;
        config.rw_low_degree_value = 10000000;
        config.rw_ending_additional_settle = false;
        config.rw_repetitions_per_node = 1;
        config.naive_settle_on_insertion = false;
        config.bgs_factor = 1.0;
        config.maintain_opt = false;

        // CLI parse_parameters.h: dynblossom overrides rw_max_length
        if (config.algorithm == DYNBLOSSOM) {
            config.rw_max_length = std::numeric_limits<int>::max() / 2;
        }

        switch (config.algorithm) {
            case RANDOM_WALK:
                solver_.reset(new rw_dyn_matching(G_.get(), config));
                break;
            case BASWANA_GUPTA_SENG:
                solver_.reset(new baswanaguptasen_dyn_matching(G_.get(), config));
                break;
            case NEIMAN_SOLOMON:
                solver_.reset(new neimansolomon_dyn_matching(G_.get(), config));
                break;
            case NAIVE:
                solver_.reset(new naive_dyn_matching(G_.get(), config));
                break;
            case DYNBLOSSOM:
                solver_.reset(new blossom_dyn_matching(G_.get(), config));
                break;
            case DYNBLOSSOMNAIVE:
                solver_.reset(new blossom_dyn_matching_naive(G_.get(), config));
                break;
            case BLOSSOM:
                solver_.reset(new static_blossom(G_.get(), config));
                break;
        }

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    void insert_edge(int u, int v) {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        // The graph stores edges in both directions
        G_->new_edge(static_cast<NodeID>(u), static_cast<NodeID>(v));
        G_->new_edge(static_cast<NodeID>(v), static_cast<NodeID>(u));
        solver_->new_edge(static_cast<NodeID>(u), static_cast<NodeID>(v));

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    void delete_edge(int u, int v) {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        G_->remove_edge(static_cast<NodeID>(u), static_cast<NodeID>(v));
        G_->remove_edge(static_cast<NodeID>(v), static_cast<NodeID>(u));
        solver_->remove_edge(static_cast<NodeID>(u), static_cast<NodeID>(v));

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    int get_matching_size() {
        return static_cast<int>(solver_->getMSize());
    }

    py::array_t<int32_t> get_matching() {
        auto& M = solver_->getM();
        py::array_t<int32_t> result(num_nodes_);
        auto r = result.mutable_unchecked<1>();
        for (int i = 0; i < num_nodes_; i++) {
            NodeID mate = (i < static_cast<int>(M.size())) ? M[i] : NOMATE;
            r(i) = (mate == NOMATE) ? -1 : static_cast<int32_t>(mate);
        }
        return result;
    }

private:
    int num_nodes_;
    std::string algorithm_name_;
    std::unique_ptr<dyn_graph_access> G_;
    std::unique_ptr<dyn_matching> solver_;
};

PYBIND11_MODULE(_dyn_matching, m) {
    m.doc() = "Python bindings for DynMatch (dynamic graph matching)";

    py::class_<DynMatchingSolver>(m, "DynMatchingSolver")
        .def(py::init<int, const std::string&, int>(),
             py::arg("num_nodes"), py::arg("algorithm"), py::arg("seed"),
             R"doc(
             Create a dynamic matching solver.

             Parameters
             ----------
             num_nodes : int
                 Number of vertices.
             algorithm : str
                 Algorithm name.
             seed : int
                 Random seed.
             )doc")
        .def("insert_edge", &DynMatchingSolver::insert_edge,
             py::arg("u"), py::arg("v"),
             "Insert an undirected edge (u, v).")
        .def("delete_edge", &DynMatchingSolver::delete_edge,
             py::arg("u"), py::arg("v"),
             "Delete an undirected edge (u, v).")
        .def("get_matching_size", &DynMatchingSolver::get_matching_size,
             "Return current matching size (number of matched edges).")
        .def("get_matching", &DynMatchingSolver::get_matching,
             "Return matching array: matching[v] = mate of v, or -1.");
}
