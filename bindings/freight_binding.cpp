/**
 * freight_binding.cpp — pybind11 binding for FREIGHT streaming hypergraph partitioning.
 *
 * Provides both a one-shot partition function (from HyperGraph CSR arrays)
 * and a true streaming class (FreightPartitioner) that returns partition IDs
 * immediately per node.
 *
 * Replicates the exact code path of freight.cpp to produce bit-identical
 * results for the same configuration and input order.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

/* FREIGHT headers */
#include "definitions.h"
#include "random_functions.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"
#include "partition/onepass_partitioning/fennel.h"
#include "partition/onepass_partitioning/fennel_approx_sqrt.h"
#include "partition/onepass_partitioning/ldg.h"

namespace py = pybind11;


/* ── Algorithm / objective enums from string ── */

static int parse_algorithm(const std::string& algo) {
    if (algo == "fennel_approx_sqrt") return ONEPASS_FENNEL_APPROX_SQRT;
    if (algo == "fennel")             return ONEPASS_FENNEL;
    if (algo == "ldg")                return ONEPASS_LDG;
    if (algo == "hashing")            return ONEPASS_HASHING;
    throw std::runtime_error(
        "FREIGHT: unknown algorithm '" + algo +
        "'. Choose from: fennel_approx_sqrt, fennel, ldg, hashing");
}


/* ── Create partitioner matching freight.cpp initialize_onepass_partitioner ── */

static vertex_partitioning* create_partitioner(
    int algorithm, PartitionID k, PartitionID rec_bisection_base,
    float fennel_gamma, int sampling_type, PartitionID n_samples)
{
    std::vector<int> group_sizes;  /* empty — no hierarchy from group_sizes */
    bool given_hierarchy = false;

    switch (algorithm) {
        case ONEPASS_HASHING:
        case ONEPASS_HASHING_CRC32:
            return new vertex_partitioning(
                0, k - 1, rec_bisection_base, 1,
                given_hierarchy, group_sizes, sampling_type, n_samples, true);
        case ONEPASS_LDG:
            return new onepass_ldg(
                0, k - 1, rec_bisection_base, 1,
                given_hierarchy, group_sizes, sampling_type, n_samples, false);
        case ONEPASS_FENNEL:
            return new onepass_fennel(
                0, k - 1, rec_bisection_base, 1,
                given_hierarchy, group_sizes, sampling_type, n_samples, false, fennel_gamma);
        case ONEPASS_FENNEL_APPROX_SQRT:
        default:
            return new onepass_fennel_approx_sqrt(
                0, k - 1, rec_bisection_base, 1,
                given_hierarchy, group_sizes, sampling_type, n_samples, false, fennel_gamma);
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
 * FreightPartitioner — true streaming class
 * ══════════════════════════════════════════════════════════════════════════════ */

class FreightPartitioner {
public:
    FreightPartitioner(
        int64_t num_nodes,
        int64_t num_nets,
        int k,
        double imbalance,
        const std::string& algorithm_str,
        const std::string& objective_str,
        int seed,
        bool hierarchical,
        int rec_bisection_base,
        float fennel_gamma,
        int sampling_type,
        int n_samples,
        float sampling_threshold)
    : num_nodes_(num_nodes),
      num_nets_(num_nets),
      k_(k),
      next_net_id_(0),
      use_connectivity_(objective_str == "connectivity")
    {
        if (objective_str != "cut_net" && objective_str != "connectivity") {
            throw std::runtime_error(
                "FREIGHT: unknown objective '" + objective_str +
                "'. Choose from: cut_net, connectivity");
        }
        if (k < 2) {
            throw std::runtime_error("FREIGHT: k must be >= 2");
        }

        int algorithm = parse_algorithm(algorithm_str);

        /* Seed RNG matching freight.cpp */
        srand(seed);
        random_functions::setSeed(seed);

        /* Create partitioner (order matches freight.cpp initialize_onepass_partitioner) */
        partitioner_ = create_partitioner(
            algorithm, k, rec_bisection_base, fennel_gamma,
            sampling_type, n_samples);
        if (sampling_type == SAMPLING_INACTIVE_LINEAR_COMPLEXITY ||
            sampling_type == SAMPLING_NEIGHBORS_LINEAR_COMPLEXITY) {
            partitioner_->enable_self_sorting_array();
        }
        partitioner_->set_sampling_threashold(sampling_threshold);

        /* Instantiate blocks (matching freight.cpp line 105) */
        partitioner_->instantiate_blocks(num_nodes, num_nets, k, imbalance);

        /* Hierarchical recursive bisection */
        if (hierarchical) {
            LongNodeID n = num_nodes;
            LongEdgeID m = num_nets;
            PartitionID kk = k;
            bool given_hierarchy = false;
            bool orig_alpha = false;
            PartitionID non_hash_layers = k;  /* default: all layers non-hash */
            partitioner_->create_problem_tree(n, m, kk, given_hierarchy, orig_alpha, non_hash_layers);
        }

        /* Allocate state vectors */
        stream_nodes_assign_.resize(num_nodes, INVALID_PARTITION);
        stream_edges_assign_.resize(num_nets, INVALID_PARTITION);
        stream_blocks_weight_.resize(k, 0);

        /* Thread-local state (single-threaded) */
        neighbor_blocks_.resize(k);
        all_blocks_to_keys_.resize(k, INVALID_PARTITION);
        next_key_ = 0;
    }

    ~FreightPartitioner() {
        delete partitioner_;
    }

    /* Assign a single node, return its partition block ID immediately */
    int assign_node(
        int64_t node_id,
        const std::vector<std::vector<int64_t>>& nets,
        const std::vector<int64_t>& net_weights,
        int64_t node_weight)
    {
        if (node_id < 0 || node_id >= num_nodes_) {
            throw std::runtime_error("FREIGHT: node_id out of range");
        }

        int algorithm = -1;  /* detect hashing from partitioner */
        int my_thread = 0;

        /* Resolve net vertex lists → internal net IDs */
        std::vector<int64_t> resolved_net_ids;
        resolved_net_ids.reserve(nets.size());
        for (const auto& net_verts : nets) {
            resolved_net_ids.push_back(get_or_create_net_id(net_verts));
        }

        /* Clear edge weights on previously used neighbor blocks */
        partitioner_->clear_edgeweight_blocks(neighbor_blocks_, next_key_, my_thread);
        next_key_ = 0;

        /* Accumulate neighbor block weights from nets
         * (matching readNodeOnePass_netl lines 274-293, no sampling path) */
        valid_neighboring_nets_.clear();
        for (size_t i = 0; i < resolved_net_ids.size(); i++) {
            int64_t net_id = resolved_net_ids[i];
            EdgeWeight edge_weight = (i < net_weights.size()) ? net_weights[i] : 1;

            PartitionID target_block = stream_edges_assign_[net_id];

            if (target_block != CUT_NET) {
                valid_neighboring_nets_.push_back(net_id);
            }

            if (target_block != INVALID_PARTITION && target_block != CUT_NET) {
                PartitionID key = all_blocks_to_keys_[target_block];
                if (key >= next_key_ || neighbor_blocks_[key].first != target_block) {
                    all_blocks_to_keys_[target_block] = next_key_;
                    auto& new_element = neighbor_blocks_[next_key_];
                    new_element.first = target_block;
                    new_element.second = edge_weight;
                    next_key_++;
                } else {
                    neighbor_blocks_[key].second += edge_weight;
                }
            }
        }

        /* Load accumulated edges into partitioner (matching line 296-298) */
        for (PartitionID key = 0; key < next_key_; key++) {
            auto& element = neighbor_blocks_[key];
            partitioner_->load_edge(element.first, element.second, my_thread);
        }

        /* Solve: assign node to best block (matching line 147) */
        PartitionID block = partitioner_->solve_node(
            static_cast<LongNodeID>(node_id),
            static_cast<NodeWeight>(node_weight),
            my_thread);

        /* Register result (matching register_result lines 343-356) */
        stream_nodes_assign_[node_id] = block;
        stream_blocks_weight_[block] += 1;

        /* Update per-net block tracking */
        for (auto& net_id : valid_neighboring_nets_) {
            PartitionID& old_block = stream_edges_assign_[net_id];
            if (use_connectivity_) {
                old_block = block;
            } else {
                /* cut-net logic */
                old_block = (old_block == block || old_block == INVALID_PARTITION)
                    ? block : CUT_NET;
            }
        }

        return static_cast<int>(block);
    }

    py::array_t<int> get_assignment() {
        py::array_t<int> result(num_nodes_);
        auto r = result.mutable_unchecked<1>();
        for (int64_t i = 0; i < num_nodes_; i++) {
            r(i) = static_cast<int>(stream_nodes_assign_[i]);
        }
        return result;
    }

private:
    int64_t get_or_create_net_id(const std::vector<int64_t>& net_verts) {
        /* Sort the vertex list to use as key */
        std::vector<int64_t> sorted_verts(net_verts);
        std::sort(sorted_verts.begin(), sorted_verts.end());

        auto it = net_map_.find(sorted_verts);
        if (it != net_map_.end()) {
            return it->second;
        }

        int64_t id = next_net_id_++;
        if (id >= num_nets_) {
            throw std::runtime_error(
                "FREIGHT: discovered more nets than num_nets=" +
                std::to_string(num_nets_) + ". Increase num_nets.");
        }
        net_map_[sorted_verts] = id;
        return id;
    }

    int64_t num_nodes_;
    int64_t num_nets_;
    int k_;

    vertex_partitioning* partitioner_;

    std::vector<PartitionID> stream_nodes_assign_;
    std::vector<PartitionID> stream_edges_assign_;
    std::vector<NodeWeight> stream_blocks_weight_;

    /* Net identification: sorted vertex list → internal net ID */
    std::map<std::vector<int64_t>, int64_t> net_map_;
    int64_t next_net_id_;

    /* Thread-local state (single-threaded) */
    std::vector<std::pair<PartitionID, EdgeWeight>> neighbor_blocks_;
    std::vector<PartitionID> all_blocks_to_keys_;
    PartitionID next_key_;

    /* Valid neighboring nets for register_result */
    std::vector<int64_t> valid_neighboring_nets_;

    bool use_connectivity_;
};


/* ══════════════════════════════════════════════════════════════════════════════
 * One-shot function: partition from HyperGraph CSR arrays
 * ══════════════════════════════════════════════════════════════════════════════ */

static py::array_t<int> freight_partition(
    const py::array_t<int64_t>& vptr,
    const py::array_t<int32_t>& vedges,
    int64_t num_nets,
    const py::array_t<int64_t>& node_weights,
    const py::array_t<int64_t>& edge_weights,
    int k,
    double imbalance,
    const std::string& algorithm_str,
    const std::string& objective_str,
    int seed,
    int num_streams_passes,
    bool hierarchical,
    int rec_bisection_base,
    float fennel_gamma,
    int sampling_type,
    int n_samples,
    float sampling_threshold,
    bool suppress_output)
{
    auto vp = vptr.unchecked<1>();
    auto ve = vedges.unchecked<1>();
    int64_t n = vp.shape(0) - 1;

    bool has_node_weights = (node_weights.size() > 0);
    bool has_edge_weights = (edge_weights.size() > 0);

    const int64_t* nw_ptr = has_node_weights ? node_weights.data() : nullptr;
    const int64_t* ew_ptr = has_edge_weights ? edge_weights.data() : nullptr;

    bool use_connectivity = (objective_str == "connectivity");

    /* Suppress output */
    std::streambuf* saved_cout = nullptr;
    std::streambuf* saved_cerr = nullptr;
    std::ofstream devnull;
    if (suppress_output) {
        devnull.open("/dev/null");
        saved_cout = std::cout.rdbuf(devnull.rdbuf());
        saved_cerr = std::cerr.rdbuf(devnull.rdbuf());
    }
    auto restore_output = [&]() {
        if (suppress_output) {
            std::cout.rdbuf(saved_cout);
            std::cerr.rdbuf(saved_cerr);
        }
    };

    try {
        int algorithm = parse_algorithm(algorithm_str);
        if (objective_str != "cut_net" && objective_str != "connectivity") {
            throw std::runtime_error(
                "FREIGHT: unknown objective '" + objective_str +
                "'. Choose from: cut_net, connectivity");
        }

        /* Seed RNG */
        srand(seed);
        random_functions::setSeed(seed);

        /* Create partitioner (order matches freight.cpp initialize_onepass_partitioner) */
        vertex_partitioning* partitioner = create_partitioner(
            algorithm, k, rec_bisection_base, fennel_gamma,
            sampling_type, n_samples);
        if (sampling_type == SAMPLING_INACTIVE_LINEAR_COMPLEXITY ||
            sampling_type == SAMPLING_NEIGHBORS_LINEAR_COMPLEXITY) {
            partitioner->enable_self_sorting_array();
        }
        partitioner->set_sampling_threashold(sampling_threshold);

        bool use_self_sorting_array =
            (sampling_type == SAMPLING_INACTIVE_LINEAR_COMPLEXITY ||
             sampling_type == SAMPLING_NEIGHBORS_LINEAR_COMPLEXITY);

        /* Allocate state */
        std::vector<PartitionID> stream_nodes_assign(n, INVALID_PARTITION);
        std::vector<PartitionID> stream_edges_assign(num_nets, INVALID_PARTITION);
        std::vector<NodeWeight> stream_blocks_weight(k, 0);

        /* Thread-local state */
        std::vector<std::pair<PartitionID, EdgeWeight>> neighbor_blocks(k);
        std::vector<PartitionID> all_blocks_to_keys(k, INVALID_PARTITION);
        PartitionID next_key = 0;
        std::vector<int64_t> valid_neighboring_nets;

        /* Build edge-to-vertex mapping for evaluation (only needed for multi-pass) */
        std::vector<std::vector<int64_t>> net_to_nodes;
        if (num_streams_passes > 1) {
            net_to_nodes.resize(num_nets);
            for (int64_t node = 0; node < n; node++) {
                for (int64_t e = vp(node); e < vp(node + 1); e++) {
                    net_to_nodes[ve(e)].push_back(node);
                }
            }
        }

        /* Best partition tracking for restreaming */
        std::vector<PartitionID> best_nodes_assign;
        std::vector<NodeWeight> best_blocks_weight;
        double best_objective = std::numeric_limits<double>::max();

        /* Multi-pass streaming */
        for (int pass = 0; pass < num_streams_passes; pass++) {
            /* Instantiate blocks (only on first pass due to size > 0 guard) */
            partitioner->instantiate_blocks(n, num_nets, k, imbalance);
            if (pass > 0 && use_self_sorting_array) {
                partitioner->reset_sorted_blocks();
            }

            /* Hierarchical mode */
            if (hierarchical && pass == 0) {
                LongNodeID nn = n;
                LongEdgeID mm = num_nets;
                PartitionID kk = k;
                partitioner->create_problem_tree(nn, mm, kk, false, false, kk);
            }

            /* Reset thread-local state each pass (matching CLI's OpenMP init block) */
            std::fill(all_blocks_to_keys.begin(), all_blocks_to_keys.end(), INVALID_PARTITION);
            next_key = 0;

            /* Restreaming: reset CUT_NET entries in cut-net mode so
               previously-cut nets can be reconsidered (matches readFirstLineStream) */
            if (pass > 0 && !use_connectivity) {
                for (auto& entry : stream_edges_assign) {
                    if (entry == CUT_NET) entry = INVALID_PARTITION;
                }
            }

            /* Process all nodes */
            for (int64_t curr_node = 0; curr_node < n; curr_node++) {
                int my_thread = 0;

                NodeWeight nw_val = has_node_weights ? static_cast<NodeWeight>(nw_ptr[curr_node]) : 1;

                /* Restreaming: remove vertex from its old block before re-evaluating */
                if (pass > 0) {
                    PartitionID old_block = stream_nodes_assign[curr_node];
                    if (old_block != INVALID_PARTITION) {
                        stream_blocks_weight[old_block] -= 1;
                        partitioner->remove_nodeweight(old_block, 1);
                        if (use_self_sorting_array) {
                            partitioner->decrement_sorted_block(old_block);
                        }
                    }
                }

                /* Skip I/O for hashing */
                if (algorithm == ONEPASS_HASHING || algorithm == ONEPASS_HASHING_CRC32) {
                    PartitionID block = partitioner->solve_node(curr_node, nw_val, my_thread);
                    stream_nodes_assign[curr_node] = block;
                    stream_blocks_weight[block] += 1;
                    continue;
                }

                /* Clear edge weights */
                partitioner->clear_edgeweight_blocks(neighbor_blocks, next_key, my_thread);
                next_key = 0;
                valid_neighboring_nets.clear();

                /* Read node's nets from vptr/vedges CSR */
                int64_t edge_begin = vp(curr_node);
                int64_t edge_end = vp(curr_node + 1);

                for (int64_t e = edge_begin; e < edge_end; e++) {
                    int64_t net_id = ve(e);  /* 0-indexed net ID */
                    EdgeWeight edge_wt = has_edge_weights ? static_cast<EdgeWeight>(ew_ptr[net_id]) : 1;

                    PartitionID target_block = stream_edges_assign[net_id];

                    if (target_block != CUT_NET) {
                        valid_neighboring_nets.push_back(net_id);
                    }

                    if (target_block != INVALID_PARTITION && target_block != CUT_NET) {
                        PartitionID key = all_blocks_to_keys[target_block];
                        if (key >= next_key || neighbor_blocks[key].first != target_block) {
                            all_blocks_to_keys[target_block] = next_key;
                            auto& new_element = neighbor_blocks[next_key];
                            new_element.first = target_block;
                            new_element.second = edge_wt;
                            next_key++;
                        } else {
                            neighbor_blocks[key].second += edge_wt;
                        }
                    }
                }

                /* Load edges */
                for (PartitionID key = 0; key < next_key; key++) {
                    auto& element = neighbor_blocks[key];
                    partitioner->load_edge(element.first, element.second, my_thread);
                }

                /* Solve */
                PartitionID block = partitioner->solve_node(curr_node, nw_val, my_thread);

                /* Register result */
                stream_nodes_assign[curr_node] = block;
                stream_blocks_weight[block] += 1;

                /* Update per-net tracking */
                for (auto& net_id : valid_neighboring_nets) {
                    PartitionID& old_block = stream_edges_assign[net_id];
                    if (use_connectivity) {
                        old_block = block;
                    } else {
                        old_block = (old_block == block || old_block == INVALID_PARTITION)
                            ? block : CUT_NET;
                    }
                }
            }

            /* Evaluate this pass and track best partition */
            if (num_streams_passes > 1) {
                std::vector<PartitionID> saved_edges_assign = stream_edges_assign;

                double pass_cut = 0, pass_con = 0;
                for (int64_t net = 0; net < num_nets; net++) {
                    std::set<PartitionID> blocks_in_net;
                    for (auto node : net_to_nodes[net]) {
                        blocks_in_net.insert(stream_nodes_assign[node]);
                    }
                    if (blocks_in_net.size() > 1) {
                        pass_cut += 1;
                        pass_con += blocks_in_net.size() - 1;
                    }
                }

                double pass_objective = use_connectivity ? pass_con : pass_cut;
                if (pass_objective < best_objective) {
                    best_objective = pass_objective;
                    best_nodes_assign = stream_nodes_assign;
                    best_blocks_weight = stream_blocks_weight;
                }

                stream_edges_assign = saved_edges_assign;
            }
        }

        /* Restore best partition if restreaming was used */
        if (num_streams_passes > 1 && !best_nodes_assign.empty()) {
            stream_nodes_assign = best_nodes_assign;
            stream_blocks_weight = best_blocks_weight;
        }

        /* Build result */
        py::array_t<int> result(n);
        auto r = result.mutable_unchecked<1>();
        for (int64_t i = 0; i < n; i++) {
            r(i) = static_cast<int>(stream_nodes_assign[i]);
        }

        delete partitioner;
        restore_output();
        return result;

    } catch (...) {
        restore_output();
        throw;
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
 * pybind11 module definition
 * ══════════════════════════════════════════════════════════════════════════════ */

PYBIND11_MODULE(_freight, m) {
    m.doc() = "FREIGHT streaming hypergraph partitioning";

    /* One-shot function */
    m.def("freight_partition", &freight_partition,
          py::arg("vptr"),
          py::arg("vedges"),
          py::arg("num_nets"),
          py::arg("node_weights"),
          py::arg("edge_weights"),
          py::arg("k") = 2,
          py::arg("imbalance") = 3.0,
          py::arg("algorithm") = "fennel_approx_sqrt",
          py::arg("objective") = "cut_net",
          py::arg("seed") = 0,
          py::arg("num_streams_passes") = 1,
          py::arg("hierarchical") = false,
          py::arg("rec_bisection_base") = 2,
          py::arg("fennel_gamma") = 1.5f,
          py::arg("sampling_type") = (int)SAMPLING_INACTIVE_LINEAR_COMPLEXITY,
          py::arg("n_samples") = 0,
          py::arg("sampling_threshold") = 1.0f,
          py::arg("suppress_output") = true,
          "Run FREIGHT streaming hypergraph partitioning on CSR arrays.\n\n"
          "Parameters:\n"
          "  vptr, vedges   : vertex-to-edge CSR (0-indexed)\n"
          "  num_nets       : total number of hyperedges\n"
          "  node_weights   : node weights (empty = all 1s)\n"
          "  edge_weights   : edge weights (empty = all 1s)\n"
          "  k              : number of partitions\n"
          "  imbalance      : allowed imbalance in percent\n"
          "  algorithm      : fennel_approx_sqrt, fennel, ldg, hashing\n"
          "  objective      : cut_net, connectivity\n"
          "  seed           : random seed\n"
          "  num_streams_passes : number of streaming passes\n"
          "  hierarchical   : enable recursive bisection\n\n"
          "Returns:\n"
          "  assignment array (partition ID per node)\n"
    );

    /* Streaming class */
    py::class_<FreightPartitioner>(m, "FreightPartitioner")
        .def(py::init<int64_t, int64_t, int, double,
                       const std::string&, const std::string&,
                       int, bool, int, float, int, int, float>(),
             py::arg("num_nodes"),
             py::arg("num_nets"),
             py::arg("k") = 2,
             py::arg("imbalance") = 3.0,
             py::arg("algorithm") = "fennel_approx_sqrt",
             py::arg("objective") = "cut_net",
             py::arg("seed") = 0,
             py::arg("hierarchical") = false,
             py::arg("rec_bisection_base") = 2,
             py::arg("fennel_gamma") = 1.5f,
             py::arg("sampling_type") = (int)SAMPLING_INACTIVE_LINEAR_COMPLEXITY,
             py::arg("n_samples") = 0,
             py::arg("sampling_threshold") = 1.0f)
        .def("assign_node", &FreightPartitioner::assign_node,
             py::arg("node_id"),
             py::arg("nets"),
             py::arg("net_weights") = std::vector<int64_t>(),
             py::arg("node_weight") = 1,
             "Assign a single node to a partition block.\n\n"
             "Parameters:\n"
             "  node_id     : node ID (0-indexed)\n"
             "  nets        : list of nets (each net is a list of vertex IDs)\n"
             "  net_weights : optional weight per net\n"
             "  node_weight : weight of this node\n\n"
             "Returns: assigned partition block ID\n")
        .def("get_assignment", &FreightPartitioner::get_assignment,
             "Return the partition assignment array.");
}
