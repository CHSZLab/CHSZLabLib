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
#include "clustering_evolutionary/evolutionary_signed_graph_clusterer.h"
#include "quality_metrics.h"
#include "random_functions.h"

namespace py = pybind11;

static std::tuple<int, int, py::array_t<int>>
py_evolutionary_correlation_clustering(
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

    // Configure for evolutionary clustering
    PartitionConfig partition_config;
    configuration cfg;
    cfg.clustering_evolutionary(partition_config);
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

    // Run evolutionary clustering (pseudo-MPI: single process)
    evolutionary_signed_graph_clusterer evo;
    evo.perform_evolutionary_signed_clustering(partition_config, G);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    // Extract results
    quality_metrics qm;
    int edge_cut = static_cast<int>(qm.edge_cut(G));

    // Renumber clusters contiguously
    int num_clusters = 0;
    std::vector<bool> used(G.number_of_nodes(), false);
    std::vector<PartitionID> remap(G.number_of_nodes(), 0);
    forall_nodes(G, node) {
        PartitionID p = G.getPartitionIndex(node);
        if (!used[p]) {
            used[p] = true;
            remap[p] = num_clusters++;
        }
    } endfor

    py::array_t<int> assignment(n);
    auto r = assignment.mutable_unchecked<1>();
    forall_nodes(G, node) {
        r(node) = static_cast<int>(remap[G.getPartitionIndex(node)]);
    } endfor

    return std::make_tuple(edge_cut, num_clusters, assignment);
}

PYBIND11_MODULE(_scc_evo, m) {
    m.doc() = "Python bindings for evolutionary ScalableCorrelationClustering";

    m.def("evolutionary_correlation_clustering",
          &py_evolutionary_correlation_clustering,
          py::arg("xadj"), py::arg("adjncy"), py::arg("adjwgt"),
          py::arg("vwgt"), py::arg("seed"), py::arg("time_limit"),
          R"doc(
          Run memetic evolutionary signed graph correlation clustering.

          Returns
          -------
          tuple[int, int, ndarray[int32]]
              (edge_cut, num_clusters, assignment).
          )doc");
}
