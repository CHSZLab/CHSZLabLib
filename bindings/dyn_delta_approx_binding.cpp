#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Minimal DynGraph shim satisfying the template requirements.
// Must be defined before including algorithm headers.
struct SimpleDynGraph {
    using NodeType = int;
    using StreamMember = std::pair<int, int>;
    size_t n_nodes;
    size_t n_edges;
    SimpleDynGraph(size_t nodes, size_t edges) : n_nodes(nodes), n_edges(edges) {}
    size_t currentNumEdges() const { return n_edges; }
    size_t currentNumNodes() const { return n_nodes; }
};

// Include the actual algorithm headers from DynDeltaApprox.
// Stub headers for absl/ and dyn_algorithm_impl.h are resolved via
// the dyn_delta_approx_stubs/ include directory.
#include "app/algorithms/dynamic/edge_orientation/cchhqrs.h"
#include "app/algorithms/dynamic/edge_orientation/limited_bfs.h"
#include "app/algorithms/dynamic/edge_orientation/strong_bfs.h"
#include "app/algorithms/dynamic/edge_orientation/improved_bfs.h"
#include "app/algorithms/dynamic/edge_orientation/packed_cchhqrs.h"

namespace py = pybind11;

// Convenience aliases for the template instantiations
using CCHHQRS = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::CCHHQRSEdgeOrientation<SimpleDynGraph>;
using LimitedBFS = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::LimitedBFSEdgeOrientation<SimpleDynGraph>;
using StrongBFS = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::StrongEdgeOrientation<SimpleDynGraph>;
using ImprovedBFS = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::ImprovedEdgeOrientation<SimpleDynGraph>;

using PackedVector = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::PackedCCHHQRSEdgeOrientation<
    SimpleDynGraph,
    dyn_delta_approx::app::algorithms::dynamic::edge_orientation::InBucketsVector>;
using PackedList = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::PackedCCHHQRSEdgeOrientation<
    SimpleDynGraph,
    dyn_delta_approx::app::algorithms::dynamic::edge_orientation::InBucketsList>;
using PackedMap = dyn_delta_approx::app::algorithms::dynamic::edge_orientation::PackedCCHHQRSEdgeOrientation<
    SimpleDynGraph,
    dyn_delta_approx::app::algorithms::dynamic::edge_orientation::InBucketsMap>;

// Type-erased wrapper so we can pick algorithm at runtime
class DynDeltaApproxSolver {
public:
    DynDeltaApproxSolver(int num_nodes, int num_edges_hint,
                         const std::string& algorithm,
                         double lambda_param, int theta, int b,
                         int bfs_depth)
        : num_nodes_(num_nodes), algorithm_name_(algorithm)
    {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        SimpleDynGraph g(num_nodes, num_edges_hint);

        std::map<std::string, double> dparams;
        dparams["lambda"] = lambda_param;
        std::map<std::string, int64_t> iparams;
        iparams["theta"] = theta;
        iparams["b"] = b;
        iparams["bfs_depth"] = bfs_depth;

        // Use reset(new ...) instead of make_unique because the constructors
        // are templated on MapType/StringType and Clang cannot deduce those
        // template parameters through make_unique's perfect forwarding.
        if (algorithm == "cchhqrs") {
            algo_cchhqrs_.reset(new CCHHQRS(g, dparams, iparams));
            which_ = ALG_CCHHQRS;
        } else if (algorithm == "limited_bfs") {
            algo_limited_bfs_.reset(new LimitedBFS(g, dparams, iparams));
            which_ = ALG_LIMITED_BFS;
        } else if (algorithm == "strong_bfs") {
            algo_strong_bfs_.reset(new StrongBFS(g, dparams, iparams));
            which_ = ALG_STRONG_BFS;
        } else if (algorithm == "improved_bfs") {
            algo_improved_bfs_.reset(new ImprovedBFS(g, dparams, iparams));
            which_ = ALG_IMPROVED_BFS;
        } else if (algorithm == "packed_cchhqrs") {
            algo_packed_vector_.reset(new PackedVector(g, dparams, iparams));
            which_ = ALG_PACKED_VECTOR;
        } else if (algorithm == "packed_cchhqrs_list") {
            algo_packed_list_.reset(new PackedList(g, dparams, iparams));
            which_ = ALG_PACKED_LIST;
        } else if (algorithm == "packed_cchhqrs_map") {
            algo_packed_map_.reset(new PackedMap(g, dparams, iparams));
            which_ = ALG_PACKED_MAP;
        } else {
            std::cout.rdbuf(old_cout);
            std::cerr.rdbuf(old_cerr);
            throw std::invalid_argument("Unknown algorithm: " + algorithm);
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

        SimpleDynGraph::StreamMember edge{u, v};
        switch (which_) {
            case ALG_CCHHQRS:       algo_cchhqrs_->handle_insertion(edge); break;
            case ALG_LIMITED_BFS:   algo_limited_bfs_->handle_insertion(edge); break;
            case ALG_STRONG_BFS:    algo_strong_bfs_->handle_insertion(edge); break;
            case ALG_IMPROVED_BFS:  algo_improved_bfs_->handle_insertion(edge); break;
            case ALG_PACKED_VECTOR: algo_packed_vector_->handle_insertion(edge); break;
            case ALG_PACKED_LIST:   algo_packed_list_->handle_insertion(edge); break;
            case ALG_PACKED_MAP:    algo_packed_map_->handle_insertion(edge); break;
        }

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    void delete_edge(int u, int v) {
        std::streambuf* old_cout = std::cout.rdbuf();
        std::streambuf* old_cerr = std::cerr.rdbuf();
        std::ostringstream null_stream;
        std::cout.rdbuf(null_stream.rdbuf());
        std::cerr.rdbuf(null_stream.rdbuf());

        SimpleDynGraph::StreamMember edge{u, v};
        switch (which_) {
            case ALG_CCHHQRS:       algo_cchhqrs_->handle_deletion(edge); break;
            case ALG_LIMITED_BFS:   algo_limited_bfs_->handle_deletion(edge); break;
            case ALG_STRONG_BFS:    algo_strong_bfs_->handle_deletion(edge); break;
            case ALG_IMPROVED_BFS:  algo_improved_bfs_->handle_deletion(edge); break;
            case ALG_PACKED_VECTOR: algo_packed_vector_->handle_deletion(edge); break;
            case ALG_PACKED_LIST:   algo_packed_list_->handle_deletion(edge); break;
            case ALG_PACKED_MAP:    algo_packed_map_->handle_deletion(edge); break;
        }

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    int get_max_out_degree() {
        switch (which_) {
            case ALG_CCHHQRS:       return static_cast<int>(algo_cchhqrs_->size());
            case ALG_LIMITED_BFS:   return static_cast<int>(algo_limited_bfs_->size());
            case ALG_STRONG_BFS:    return static_cast<int>(algo_strong_bfs_->size());
            case ALG_IMPROVED_BFS:  return static_cast<int>(algo_improved_bfs_->size());
            case ALG_PACKED_VECTOR: return static_cast<int>(algo_packed_vector_->size());
            case ALG_PACKED_LIST:   return static_cast<int>(algo_packed_list_->size());
            case ALG_PACKED_MAP:    return static_cast<int>(algo_packed_map_->size());
        }
        return 0;
    }

private:
    int num_nodes_;
    std::string algorithm_name_;

    enum AlgType {
        ALG_CCHHQRS, ALG_LIMITED_BFS, ALG_STRONG_BFS, ALG_IMPROVED_BFS,
        ALG_PACKED_VECTOR, ALG_PACKED_LIST, ALG_PACKED_MAP
    } which_;

    std::unique_ptr<CCHHQRS> algo_cchhqrs_;
    std::unique_ptr<LimitedBFS> algo_limited_bfs_;
    std::unique_ptr<StrongBFS> algo_strong_bfs_;
    std::unique_ptr<ImprovedBFS> algo_improved_bfs_;
    std::unique_ptr<PackedVector> algo_packed_vector_;
    std::unique_ptr<PackedList> algo_packed_list_;
    std::unique_ptr<PackedMap> algo_packed_map_;
};

PYBIND11_MODULE(_dyn_delta_approx, m) {
    m.doc() = "Python bindings for DynDeltaApprox (dynamic edge orientation, approximate)";

    py::class_<DynDeltaApproxSolver>(m, "DynDeltaApproxSolver")
        .def(py::init<int, int, const std::string&, double, int, int, int>(),
             py::arg("num_nodes"), py::arg("num_edges_hint"),
             py::arg("algorithm"), py::arg("lambda_param"),
             py::arg("theta"), py::arg("b"), py::arg("bfs_depth"),
             R"doc(
             Create a dynamic approximate edge orientation solver.

             Parameters
             ----------
             num_nodes : int
                 Number of vertices.
             num_edges_hint : int
                 Hint for maximum number of edges (used for memory pre-allocation).
             algorithm : str
                 Algorithm name.
             lambda_param : float
                 Lambda parameter for CCHHQRS variants.
             theta : int
                 Theta parameter for CCHHQRS variants.
             b : int
                 Fractional edge parameter for CCHHQRS variants.
             bfs_depth : int
                 BFS depth for BFS-based algorithms.
             )doc")
        .def("insert_edge", &DynDeltaApproxSolver::insert_edge,
             py::arg("u"), py::arg("v"),
             "Insert an undirected edge (u, v).")
        .def("delete_edge", &DynDeltaApproxSolver::delete_edge,
             py::arg("u"), py::arg("v"),
             "Delete an undirected edge (u, v).")
        .def("get_max_out_degree", &DynDeltaApproxSolver::get_max_out_degree,
             "Return the current maximum out-degree.");
}
