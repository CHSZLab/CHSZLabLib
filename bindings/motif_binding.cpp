#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "data_structure/graph_access.h"
#include "partition/partition_config.h"
#include "configuration.h"
#include "balance_configuration.h"
#include "algorithms/bfs_depth.h"
#include "algorithms/triangle_listing.h"
#include "algorithms/mqi.h"
#include "partition/graph_partitioner.h"
#include "partition/uncoarsening/refinement/label_propagation_refinement/label_propagation_refinement.h"
#include "quality_metrics.h"
#include "random_functions.h"
#include "timer.h"

namespace py = pybind11;

struct OutputSuppressor {
    std::streambuf *old_cout, *old_cerr;
    std::ostringstream sink_out, sink_err;
    OutputSuppressor()
        : old_cout(std::cout.rdbuf(sink_out.rdbuf())),
          old_cerr(std::cerr.rdbuf(sink_err.rdbuf())) {}
    ~OutputSuppressor() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
};

// Build graph_access from CSR arrays (in-place, since copy ctor is deleted).
static void build_graph(
        graph_access &G,
        py::array_t<int, py::array::c_style> xadj,
        py::array_t<int, py::array::c_style> adjncy) {

    int n = static_cast<int>(xadj.size() - 1);
    int *xadj_ptr = const_cast<int*>(xadj.data());
    int *adjncy_ptr = const_cast<int*>(adjncy.data());
    G.build_from_metis(n, xadj_ptr, adjncy_ptr);
}

// SOCIAL: BFS ball → triangle enumeration → quotient graph → MQI flow refinement.
// Returns (cluster_nodes, motif_conductance).
static std::tuple<py::array_t<int>, double>
py_motif_cluster_social(
        py::array_t<int, py::array::c_style> xadj,
        py::array_t<int, py::array::c_style> adjncy,
        int seed_node,
        std::vector<short int> bfs_depths,
        int time_limit,
        int seed) {

    int n = static_cast<int>(xadj.size() - 1);
    if (n == 0 || bfs_depths.empty()) {
        return std::make_tuple(py::array_t<int>(0), 1.0);
    }

    OutputSuppressor suppressor;

    graph_access G;
    build_graph(G, xadj, adjncy);

    // Configure
    PartitionConfig partition_config;
    configuration cfg;
    cfg.standard(partition_config);
    partition_config.k = 2;
    partition_config.seed = seed;
    partition_config.seed_node = seed_node;
    partition_config.bfsDepths = bfs_depths;
    partition_config.fix_seed_node = true;
    partition_config.repetition_timelimit = time_limit;
    partition_config.triangle_count = -1; // UNDEFINED_TRIANGLE_COUNT

    srand(seed + seed_node);
    random_functions::setSeed(seed + seed_node);

    // Algorithm: iterate over BFS depths, keep best motif conductance
    bfs_depth bfs;
    triangle_listing triangle;
    mqi mqi_flow;

    double min_motif_conductance = 100.0;
    std::vector<NodeID> best_community;
    bool was_over_100 = false;

    std::vector<NodeID> mapping(G.number_of_nodes(), -1);
    std::vector<NodeID> *subgraph_map = NULL;
    std::vector<bool> lastLayerNodes;
    graph_access *G_temp = NULL;

    std::vector<short int> level(G.number_of_nodes(), -1);
    std::vector<bool> visited(G.number_of_nodes(), false);
    std::vector<bool> touched(G.number_of_nodes(), false);

    timer t;

    for (int alpha = 0; alpha < (int)bfs_depths.size() &&
            (alpha == 0 || t.elapsed() < time_limit); alpha++) {

        if (G_temp != NULL) delete G_temp;
        if (subgraph_map != NULL) {
            for (const auto &j : *subgraph_map) {
                lastLayerNodes[mapping[j]] = false;
                mapping[j] = -1;
            }
            delete subgraph_map;
        }

        bool force_over_100 = (alpha == (int)bfs_depths.size() - 1 && !was_over_100);
        std::vector<NodeID> seed_nodes_vec = {(NodeID)seed_node};
        G_temp = bfs.runBFS_graph(G, seed_nodes_vec, bfs_depths[alpha],
                mapping, subgraph_map, lastLayerNodes,
                level, visited, touched, force_over_100);

        long long num_triangles = triangle.triangle_run_graph(*G_temp, lastLayerNodes);

        NodeID subgraph_node_count = G_temp->number_of_nodes();
        if (subgraph_node_count > 100) was_over_100 = true;

        // Build quotient graph (SOCIAL-style: one node per non-last-layer node)
        int index = 1;
        std::vector<NodeID> map_model_original;
        map_model_original.push_back((*subgraph_map)[0]);

        forall_nodes((*G_temp), node) {
            if (node == 0) continue;
            if (lastLayerNodes[node]) {
                G_temp->setPartitionIndex(node, index);
                index++;
            } else {
                G_temp->setPartitionIndex(node, index);
                index++;
                map_model_original.push_back((*subgraph_map)[node]);
            }
        } endfor
        G_temp->set_partition_count(index);

        graph_access S;
        complete_boundary boundary(G_temp);
        boundary.fastComputeQuotientGraphRemoveZeroEdges(S, index);
        S.setNodeWeight(1, 1);

        // Set initial partition: last-layer nodes in block 1, rest in block 0
        forall_nodes(S, sn) {
            if (lastLayerNodes[sn]) {
                S.setPartitionIndex(sn, 1);
            } else {
                S.setPartitionIndex(sn, 0);
            }
        } endfor

        quality_metrics qm;
        double motif_conductance = qm.local_conductance(S, 1, partition_config.triangle_count);

        // MQI flow-based refinement
        double current_min = motif_conductance;
        int numTries = 0;
        while (numTries < 1) {
            mqi_flow.mqi_improvement_ball(S, partition_config.fix_seed_node);
            double updated = qm.local_conductance(S, 1, partition_config.triangle_count);
            if (current_min > updated) {
                current_min = updated;
                motif_conductance = updated;
            } else {
                numTries++;
            }
        }

        if (motif_conductance < min_motif_conductance) {
            min_motif_conductance = motif_conductance;
            best_community.clear();
            forall_nodes(S, sn) {
                if (S.getPartitionIndex(sn) != 1) {
                    best_community.push_back(map_model_original[sn]);
                }
            } endfor
        }
    }

    if (G_temp != NULL) delete G_temp;
    if (subgraph_map != NULL) delete subgraph_map;

    // Return cluster node IDs
    py::array_t<int> result(best_community.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < best_community.size(); i++) {
        r(i) = static_cast<int>(best_community[i]);
    }

    return std::make_tuple(result, min_motif_conductance);
}

// LMCHGP: BFS ball → triangle enumeration → quotient graph → graph partitioning.
// Returns (cluster_nodes, motif_conductance).
static std::tuple<py::array_t<int>, double>
py_motif_cluster_lmchgp(
        py::array_t<int, py::array::c_style> xadj,
        py::array_t<int, py::array::c_style> adjncy,
        int seed_node,
        std::vector<short int> bfs_depths,
        int time_limit,
        int seed) {

    int n = static_cast<int>(xadj.size() - 1);
    if (n == 0 || bfs_depths.empty()) {
        return std::make_tuple(py::array_t<int>(0), 1.0);
    }

    OutputSuppressor suppressor;

    graph_access G;
    build_graph(G, xadj, adjncy);

    // Configure
    PartitionConfig partition_config;
    configuration cfg;
    cfg.standard(partition_config);
    cfg.eco(partition_config);  // use eco quality for partitioning
    partition_config.k = 2;
    partition_config.seed = seed;
    partition_config.seed_node = seed_node;
    partition_config.bfsDepths = bfs_depths;
    partition_config.fix_seed_node = true;
    partition_config.label_prop_ls = true; // enable label propagation refinement
    partition_config.repetition_timelimit = time_limit;
    partition_config.triangle_count = -1;
    partition_config.beta = 3; // number of imbalance trials per depth

    srand(seed + seed_node);
    random_functions::setSeed(seed + seed_node);

    bfs_depth bfs;
    triangle_listing triangle;

    double min_motif_conductance = 100.0;
    std::vector<NodeID> best_community;
    bool was_over_100 = false;

    std::vector<NodeID> mapping(G.number_of_nodes(), -1);
    std::vector<NodeID> *subgraph_map = NULL;
    std::vector<bool> lastLayerNodes;
    graph_access *G_temp = NULL;

    std::vector<short int> level(G.number_of_nodes(), -1);
    std::vector<bool> visited(G.number_of_nodes(), false);
    std::vector<bool> touched(G.number_of_nodes(), false);

    timer t;

    for (int alpha = 0; alpha < (int)bfs_depths.size() &&
            (alpha == 0 || t.elapsed() < time_limit); alpha++) {

        if (G_temp != NULL) delete G_temp;
        if (subgraph_map != NULL) {
            for (const auto &j : *subgraph_map) {
                lastLayerNodes[mapping[j]] = false;
                mapping[j] = -1;
            }
            delete subgraph_map;
        }

        bool force_over_100 = (alpha == (int)bfs_depths.size() - 1 && !was_over_100);
        std::vector<NodeID> seed_nodes_vec = {(NodeID)seed_node};
        G_temp = bfs.runBFS_graph(G, seed_nodes_vec, bfs_depths[alpha],
                mapping, subgraph_map, lastLayerNodes,
                level, visited, touched, force_over_100);

        long long num_triangles = triangle.triangle_run_graph(*G_temp, lastLayerNodes);

        NodeID subgraph_node_count = G_temp->number_of_nodes();
        if (subgraph_node_count > 100) was_over_100 = true;

        // Build quotient graph (LMCHGP-style: last-layer nodes collapsed into block 1)
        int index = 2;
        std::vector<NodeID> map_model_original;
        map_model_original.push_back((*subgraph_map)[0]);
        map_model_original.push_back((*subgraph_map)[0]); // artificial node

        forall_nodes((*G_temp), node) {
            if (node == 0) continue;
            if (lastLayerNodes[node]) {
                G_temp->setPartitionIndex(node, 1);
            } else {
                G_temp->setPartitionIndex(node, index);
                index++;
                map_model_original.push_back((*subgraph_map)[node]);
            }
        } endfor
        G_temp->set_partition_count(index);

        graph_access S;
        complete_boundary boundary(G_temp);
        boundary.fastComputeQuotientGraphRemoveZeroEdges(S, index);
        S.setNodeWeight(1, 1);

        // Partition with multiple imbalance values
        for (int beta = 0; beta < partition_config.beta &&
                (beta == 0 || t.elapsed() < time_limit); beta++) {

            S.set_partition_count(2);
            balance_configuration bc;
            partition_config.imbalance = random_functions::nextInt(0, 99);
            bc.configurate_balance(partition_config, S);

            graph_partitioner partitioner;
            quality_metrics qm;

            partitioner.perform_partitioning(partition_config, S);

            S.set_partition_count(2);
            S.setPartitionIndex(0, 1 - S.getPartitionIndex(1));

            std::vector<NodeID> fixed_nodes = {0, 1};

            if (partition_config.label_prop_ls) {
                label_propagation_refinement refine;
                refine.perform_refinement_conductance(partition_config, fixed_nodes,
                        S, S.getPartitionIndex(1));
            }

            double motif_conductance = qm.local_conductance(S, S.getPartitionIndex(1),
                    partition_config.triangle_count);

            if (motif_conductance < min_motif_conductance) {
                min_motif_conductance = motif_conductance;
                best_community.clear();
                forall_nodes(S, sn) {
                    if (S.getPartitionIndex(sn) != S.getPartitionIndex(1)) {
                        best_community.push_back(map_model_original[sn]);
                    }
                } endfor
            }
        }
    }

    if (G_temp != NULL) delete G_temp;
    if (subgraph_map != NULL) delete subgraph_map;

    py::array_t<int> result(best_community.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < best_community.size(); i++) {
        r(i) = static_cast<int>(best_community[i]);
    }

    return std::make_tuple(result, min_motif_conductance);
}

PYBIND11_MODULE(_motif, m) {
    m.doc() = "Python bindings for HeidelbergMotifClustering (SOCIAL and LMCHGP)";

    m.def("motif_cluster_social", &py_motif_cluster_social,
          py::arg("xadj"), py::arg("adjncy"),
          py::arg("seed_node"), py::arg("bfs_depths"),
          py::arg("time_limit"), py::arg("seed"),
          "SOCIAL: Local motif clustering via maximum flows. "
          "Returns (cluster_node_ids, motif_conductance).");

    m.def("motif_cluster_lmchgp", &py_motif_cluster_lmchgp,
          py::arg("xadj"), py::arg("adjncy"),
          py::arg("seed_node"), py::arg("bfs_depths"),
          py::arg("time_limit"), py::arg("seed"),
          "LMCHGP: Local motif clustering via hypergraph partitioning. "
          "Returns (cluster_node_ids, motif_conductance).");
}
