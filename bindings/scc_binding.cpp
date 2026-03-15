#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

#include "data_structure/graph_access.h"
#include "partition/partition_config.h"
#include "configuration.h"
#include "clustering/signed_graph_clusterer.h"
#include "quality_metrics.h"
#include "random_functions.h"
#include "timer.h"

namespace py = pybind11;

static std::tuple<int, int, py::array_t<int>>
py_correlation_clustering(
        py::array_t<int, py::array::c_style> xadj,
        py::array_t<int, py::array::c_style> adjncy,
        py::array_t<int, py::array::c_style> adjwgt,
        py::array_t<int, py::array::c_style> vwgt,
        int seed, double time_limit) {

    int n = static_cast<int>(xadj.size() - 1);
    int *xadj_ptr = xadj.mutable_data();
    int *adjncy_ptr = adjncy.mutable_data();

    // Build graph_access from CSR
    graph_access G;
    if (vwgt.size() > 0 && adjwgt.size() > 0) {
        G.build_from_metis_weighted(n, xadj_ptr, adjncy_ptr,
                                     vwgt.mutable_data(), adjwgt.mutable_data());
    } else if (adjwgt.size() > 0) {
        std::vector<int> ones(n, 1);
        G.build_from_metis_weighted(n, xadj_ptr, adjncy_ptr,
                                     ones.data(), adjwgt.mutable_data());
    } else {
        G.build_from_metis(n, xadj_ptr, adjncy_ptr);
    }

    // Configure
    PartitionConfig partition_config;
    configuration cfg;
    cfg.clustering(partition_config);
    partition_config.seed = seed;
    partition_config.time_limit = time_limit;

    srand(seed);
    random_functions::setSeed(seed);

    // Suppress stdout/stderr
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    quality_metrics qm;
    int best_cut = std::numeric_limits<int>::max();

    if (time_limit == 0) {
        // Single run
        signed_graph_clusterer clusterer;
        clusterer.perform_signed_clustering(partition_config, G);
        best_cut = static_cast<int>(qm.edge_cut(G));
    } else {
        // Repeat within time limit, keep best
        std::vector<PartitionID> best_map(n, 0);
        PartitionID best_k = G.number_of_nodes();
        timer t;
        while (t.elapsed() < time_limit) {
            signed_graph_clusterer clusterer;
            partition_config.graph_already_partitioned = false;
            clusterer.perform_signed_clustering(partition_config, G);
            int cut = static_cast<int>(qm.edge_cut(G));
            if (cut < best_cut) {
                best_cut = cut;
                forall_nodes(G, node) {
                    best_map[node] = G.getPartitionIndex(node);
                } endfor
                best_k = G.get_partition_count();
            }
        }
        // Restore best solution
        forall_nodes(G, node) {
            G.setPartitionIndex(node, best_map[node]);
        } endfor
        partition_config.k = best_k;
        G.set_partition_count(partition_config.k);
    }

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Extract results
    int num_clusters = static_cast<int>(G.get_partition_count());
    py::array_t<int> assignment(n);
    auto r = assignment.mutable_unchecked<1>();
    forall_nodes(G, node) {
        r(node) = static_cast<int>(G.getPartitionIndex(node));
    } endfor

    return std::make_tuple(best_cut, num_clusters, assignment);
}

PYBIND11_MODULE(_scc, m) {
    m.doc() = "Python bindings for ScalableCorrelationClustering";

    m.def("correlation_clustering", &py_correlation_clustering,
          py::arg("xadj"), py::arg("adjncy"), py::arg("adjwgt"),
          py::arg("vwgt"), py::arg("seed"), py::arg("time_limit"),
          R"doc(
          Run multilevel signed graph correlation clustering.

          Returns
          -------
          tuple[int, int, ndarray[int32]]
              (edge_cut, num_clusters, assignment).
          )doc");
}
