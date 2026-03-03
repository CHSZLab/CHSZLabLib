/**
 * clustre_binding.cpp — pybind11 binding for CluStRE streaming graph clustering.
 *
 * Runs the CluStRE streaming clustering algorithm directly from CSR arrays,
 * matching the exact code paths of the CLI tool (clustering.cpp) to produce
 * identical results for the same configuration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

/* CluStRE headers */
#include "partition/partition_config.h"
#include "configuration.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"
#include "partition/onepass_partitioning/leiden.h"
#include "partition/onepass_partitioning/modularity.h"
#include "definitions.h"
#include "random_functions.h"

#include "robin_hood.h"

namespace py = pybind11;


static py::tuple py_clustre(
    const py::array_t<int64_t>& xadj,
    const py::array_t<int64_t>& adjncy,
    const std::string& mode_str,
    int seed,
    int num_streams_passes,
    double resolution_param,
    int max_num_clusters,
    int ls_time_limit,
    double ls_frac_time,
    double cut_off,
    bool suppress_output)
{
    auto xa = xadj.unchecked<1>();
    auto aj = adjncy.unchecked<1>();
    int64_t n = xa.shape(0) - 1;
    int64_t m2 = aj.shape(0);
    int64_t m  = m2 / 2;   /* undirected edges */

    /* ── 1. Initialize config (matching CLI: standard → strong → stream_map) ── */
    HeiClus::PartitionConfig config;
    {
        HeiClus::configuration cfg;
        cfg.standard(config);          /* sets all defaults */
        cfg.strong(config);            /* (calls standard again) */
        cfg.stream_map(config);        /* same as CLI with MODE_FREIGHT_GRAPHS */
    }

    config.stream_buffer_len = 1;      /* CLI sets this after stream_map */

    /* Seed random number generators (matching CLI behavior) */
    config.seed = seed;
    HeiClus::random_functions::setSeed(seed);

    config.suppress_output = true;
    config.suppress_file_output = true;
    config.stream_input = true;
    config.rle_length = -1;            /* use std::vector, not compression */
    config.cpm_gamma = resolution_param;
    config.ls_time_limit = ls_time_limit;
    config.ls_frac_time = ls_frac_time;
    config.cut_off = cut_off;
    config.ext_clustering_algorithm = NO_EXT_ALGORITHM;

    /* one_pass_algorithm: always modularity (CLI requires --one_pass_algorithm=modularity) */
    config.one_pass_algorithm = ONEPASS_MODULARITY;

    /* Override restream_amount from num_streams_passes first */
    if (num_streams_passes > 1) {
        config.restream_amount = num_streams_passes - 1;
    }

    /* Parse mode (matching CLI parse_parameters.h).
     * Mode-specific restream settings override num_streams_passes:
     * light/evo force restream_amount=0 (single pass, no restreaming). */
    if (mode_str == "light") {
        config.mode = LIGHT;
        config.restream_amount = 0;
    } else if (mode_str == "light_plus") {
        config.mode = LIGHT_PLUS;
    } else if (mode_str == "evo") {
        config.mode = EVO;
        config.restream_amount = 0;
    } else if (mode_str == "strong") {
        config.mode = STRONG;
    } else {
        throw std::runtime_error("CluStRE: unknown mode '" + mode_str +
                                 "'. Choose from: light, light_plus, evo, strong");
    }

    /* ── 2. Initialize graph metadata (matching readFirstLineStreamClustering) ── */
    config.total_nodes = n;
    config.total_edges = m;
    config.remaining_stream_nodes = n;
    config.remaining_stream_edges = m;
    config.stream_n_nodes = n;
    config.nmbNodes = 1;               /* stream_buffer_len = 1 */
    config.n_batches = n;              /* ceil(n / 1) = n */
    config.curr_batch = 0;

    if (max_num_clusters > 0) {
        config.max_num_clusters = max_num_clusters;
    } else {
        config.max_num_clusters = static_cast<int>(n * config.cluster_fraction);
    }

    /* Allocate assignment vector */
    std::vector<PartitionID> stream_nodes_assign(n, INVALID_PARTITION);
    config.stream_nodes_assign = &stream_nodes_assign;

    std::vector<NodeWeight> stream_blocks_weight;
    config.stream_blocks_weight = &stream_blocks_weight;

    config.neighbor_blocks.resize(1);
    config.all_blocks_to_keys.resize(1);
    config.next_key.resize(1);
    config.next_key[0] = 0;

    /* Initialize partitioner (matching CLI's initialize_onepass_partitioner) */
    vertex_partitioning* onepass_partitioner = nullptr;
    onepass_partitioner = new onepass_modularity(
        0, 0, config.total_nodes, 1, config.mode, false, config.cpm_gamma);

    /* Suppress stdout/stderr */
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
        /* Track total processing time for ls_frac_time check */
        auto processing_start = std::chrono::steady_clock::now();
        bool active_nodes_exist = false;

        /* ── 3. Main streaming loop (matching CLI clustering.cpp) ── */
        for (int restreaming = 0; restreaming < config.restream_amount + 1; restreaming++) {

            if (restreaming) {
                /* Reset for restreaming (matching CLI restreamingFileReset + loop init) */
                while (static_cast<int64_t>(onepass_partitioner->blocks.size()) >
                       static_cast<int64_t>(stream_blocks_weight.size())) {
                    onepass_partitioner->blocks.pop_back();
                    config.clusters_to_ix_mapping.pop_back();
                    config.neighbor_blocks[0].pop_back();
                }
                for (size_t p = 0; p < onepass_partitioner->blocks.size(); p++) {
                    onepass_partitioner->blocks[p].e_weight = 0;
                }
                config.next_key[0] = 0;
                config.remaining_stream_nodes = n;

                /* Allocate activeNodes_set on last restream pass for STRONG/LIGHT_PLUS */
                if (restreaming == config.restream_amount &&
                    (config.mode == STRONG || config.mode == LIGHT_PLUS)) {
                    config.activeNodes_set = new robin_hood::unordered_set<uint64_t>();
                }
            }

            for (int64_t curr_node = 0; curr_node < n; curr_node++) {
                int my_thread = 0;

                /* ── readNodeOnePassClustering equivalent (from CSR arrays) ── */
                auto& next_key = config.next_key[my_thread];
                auto& neighbor_blocks_vec = config.neighbor_blocks[my_thread];
                auto& clusters_to_ix_mapping = config.clusters_to_ix_mapping;

                onepass_partitioner->clear_edgeweight_blocks(neighbor_blocks_vec, next_key, my_thread);

                for (auto& nb : neighbor_blocks_vec) {
                    if (nb.first == static_cast<PartitionID>(-1)) break;
                    clusters_to_ix_mapping[nb.first] = static_cast<PartitionID>(-1);
                    nb.first = static_cast<PartitionID>(-1);
                    nb.second = 0;
                }
                onepass_partitioner->reset_streamed_edge_count();
                next_key = 0;

                /* Read neighbors from CSR arrays (0-indexed) */
                int64_t edge_begin = xa(curr_node);
                int64_t edge_end   = xa(curr_node + 1);

                for (int64_t e = edge_begin; e < edge_end; e++) {
                    int64_t neighbor = aj(e);   /* 0-indexed */
                    EdgeWeight edge_weight = 1;
                    onepass_partitioner->increment_graph_edge_count(edge_weight, restreaming);

                    PartitionID targetPar = INVALID_PARTITION;
                    if (neighbor >= 0 && neighbor < n) {
                        targetPar = stream_nodes_assign[neighbor];
                    }

                    if (targetPar != INVALID_PARTITION) {
                        PartitionID key = clusters_to_ix_mapping[targetPar];
                        if (key == static_cast<PartitionID>(-1)) {
                            clusters_to_ix_mapping[targetPar] = next_key;
                            auto& new_element = neighbor_blocks_vec[next_key];
                            new_element.first = targetPar;
                            new_element.second = edge_weight;
                            next_key++;
                        } else {
                            neighbor_blocks_vec[key].second += edge_weight;
                        }
                    }
                }

                /* Load edge weights to blocks */
                for (size_t i = 0; i < neighbor_blocks_vec.size(); i++) {
                    auto& element = neighbor_blocks_vec[i];
                    if (element.first == static_cast<PartitionID>(-1)) break;
                    onepass_partitioner->load_edge(element.first, element.second, my_thread);
                }

                config.remaining_stream_nodes--;

                /* Solve: assign node to best cluster */
                PartitionID block = onepass_partitioner->solve_node_clustering(
                    curr_node, 1, restreaming, my_thread,
                    config.neighbor_blocks[my_thread],
                    config.clusters_to_ix_mapping, config,
                    config.previous_assignment, config.kappa, false);

                /* Update quotient graph (matching CLI: only on first pass, not LIGHT/LIGHT_PLUS) */
                if (restreaming == 0 && config.mode != LIGHT && config.mode != LIGHT_PLUS) {
                    onepass_partitioner->update_quotient_graph(block, config.neighbor_blocks[my_thread]);
                }

                PartitionID orig_part = static_cast<PartitionID>(-1);
                if (restreaming) {
                    orig_part = stream_nodes_assign[curr_node];
                }
                stream_nodes_assign[curr_node] = block;

                /* Track active nodes (matching CLI: only on last restream, STRONG/LIGHT_PLUS) */
                if (restreaming == config.restream_amount
                    && orig_part != block
                    && (config.mode == STRONG || config.mode == LIGHT_PLUS)
                    && config.activeNodes_set != nullptr) {
                    active_nodes_exist = true;
                    for (int64_t e = edge_begin; e < edge_end; e++) {
                        int64_t neighbor = aj(e);
                        if (neighbor >= 0 && neighbor < n) {
                            config.activeNodes_set->insert(static_cast<uint64_t>(neighbor));
                        }
                    }
                }

                config.previous_assignment = block;
                if (static_cast<int64_t>(stream_blocks_weight.size()) <= block) {
                    stream_blocks_weight.resize(block + 1, 0);
                }
                stream_blocks_weight[block] += 1;
                if (restreaming && orig_part != static_cast<PartitionID>(-1)) {
                    stream_blocks_weight[orig_part] -= 1;
                }
            }
        }

        /* ── 4. Local search (matching CLI clustering.cpp exactly) ── */
        double ls_frac_time_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - processing_start).count();
        double ls_time_total = 0.0;

        if ((config.mode == STRONG || config.mode == LIGHT_PLUS)
            && config.activeNodes_set != nullptr
            && active_nodes_exist) {

            auto ls_start_time = std::chrono::steady_clock::now();
            std::chrono::seconds TIME_LIMIT(config.ls_time_limit);

            while (active_nodes_exist) {
                auto round_start = std::chrono::steady_clock::now();

                auto* new_active = new robin_hood::unordered_set<uint64_t>();
                onepass_partitioner->curr_round_delta_mod = 0;
                active_nodes_exist = false;

                for (size_t p = 0; p < onepass_partitioner->blocks.size(); p++) {
                    onepass_partitioner->blocks[p].e_weight = 0;
                }
                config.next_key[0] = 0;

                uint64_t previous_node = static_cast<uint64_t>(-1);
                bool timeStop = false;

                for (const auto& curr_node : *config.activeNodes_set) {
                    int my_thread = 0;

                    /* Read neighbors from CSR (O(degree) random access) */
                    auto& next_key = config.next_key[my_thread];
                    auto& neighbor_blocks_vec = config.neighbor_blocks[my_thread];
                    auto& clusters_to_ix_mapping = config.clusters_to_ix_mapping;

                    onepass_partitioner->clear_edgeweight_blocks(neighbor_blocks_vec, next_key, my_thread);
                    for (auto& nb : neighbor_blocks_vec) {
                        if (nb.first == static_cast<PartitionID>(-1)) break;
                        clusters_to_ix_mapping[nb.first] = static_cast<PartitionID>(-1);
                        nb.first = static_cast<PartitionID>(-1);
                        nb.second = 0;
                    }
                    onepass_partitioner->reset_streamed_edge_count();
                    next_key = 0;

                    int64_t edge_begin = xa(static_cast<int64_t>(curr_node));
                    int64_t edge_end   = xa(static_cast<int64_t>(curr_node) + 1);

                    for (int64_t e = edge_begin; e < edge_end; e++) {
                        int64_t neighbor = aj(e);
                        EdgeWeight edge_weight = 1;
                        onepass_partitioner->increment_graph_edge_count(edge_weight, 1);

                        PartitionID targetPar = INVALID_PARTITION;
                        if (neighbor >= 0 && neighbor < n) {
                            targetPar = stream_nodes_assign[neighbor];
                        }
                        if (targetPar != INVALID_PARTITION) {
                            PartitionID key = clusters_to_ix_mapping[targetPar];
                            if (key == static_cast<PartitionID>(-1)) {
                                clusters_to_ix_mapping[targetPar] = next_key;
                                auto& new_element = neighbor_blocks_vec[next_key];
                                new_element.first = targetPar;
                                new_element.second = edge_weight;
                                next_key++;
                            } else {
                                neighbor_blocks_vec[key].second += edge_weight;
                            }
                        }
                    }

                    for (size_t i = 0; i < neighbor_blocks_vec.size(); i++) {
                        auto& element = neighbor_blocks_vec[i];
                        if (element.first == static_cast<PartitionID>(-1)) break;
                        onepass_partitioner->load_edge(element.first, element.second, my_thread);
                    }

                    PartitionID block = onepass_partitioner->solve_node_clustering(
                        curr_node, 1, 1, my_thread,
                        config.neighbor_blocks[my_thread],
                        config.clusters_to_ix_mapping, config,
                        config.previous_assignment, config.kappa, true);

                    PartitionID orig_part = stream_nodes_assign[curr_node];
                    stream_nodes_assign[curr_node] = block;

                    if (orig_part != block) {
                        active_nodes_exist = true;
                        for (int64_t e = edge_begin; e < edge_end; e++) {
                            int64_t neighbor = aj(e);
                            if (neighbor >= 0 && neighbor < n) {
                                new_active->insert(static_cast<uint64_t>(neighbor));
                            }
                        }
                    }

                    config.previous_assignment = block;
                    if (static_cast<int64_t>(stream_blocks_weight.size()) <= block) {
                        stream_blocks_weight.resize(block + 1, 0);
                    }
                    stream_blocks_weight[block] += 1;
                    stream_blocks_weight[orig_part] -= 1;

                    /* Erase previous node from active set (matching CLI) */
                    if (previous_node != static_cast<uint64_t>(-1)) {
                        config.activeNodes_set->erase(previous_node);
                    }
                    previous_node = curr_node;

                    /* Time limit check */
                    auto elapsed = std::chrono::steady_clock::now() - ls_start_time;
                    if (elapsed > TIME_LIMIT) {
                        timeStop = true;
                        break;
                    }
                }

                delete config.activeNodes_set;
                config.activeNodes_set = new_active;
                new_active = nullptr;

                /* Convergence check (matching CLI) */
                if (onepass_partitioner->curr_round_delta_mod <
                    onepass_partitioner->overall_delta_mod * config.cut_off) {
                    break;
                }
                if (timeStop) {
                    active_nodes_exist = false;
                }

                /* ls_frac_time check (matching CLI: ls_time > ls_frac_time * total_time) */
                ls_time_total = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - ls_start_time).count();
                if (ls_time_total > config.ls_frac_time * ls_frac_time_elapsed) {
                    break;
                }

                /* Update overall delta mod */
                onepass_partitioner->overall_delta_mod += onepass_partitioner->curr_round_delta_mod;
            }
        }

        /* ── 5. Compute exact modularity (matching CLI streamEvaluateClustering) ── */
        std::vector<NodeID> blocks_weights;
        std::vector<std::pair<EdgeWeight, EdgeWeight>> blocks_sigma;
        /* blocks_sigma[c].first  = internal edges (both endpoints in cluster c) */
        /* blocks_sigma[c].second = degree sum of cluster c */

        for (int64_t node = 0; node < n; node++) {
            PartitionID partSource = stream_nodes_assign[node];

            while (static_cast<int64_t>(blocks_weights.size()) <= partSource) {
                blocks_weights.emplace_back(0);
                blocks_sigma.emplace_back(std::make_pair(0, 0));
            }
            blocks_weights[partSource]++;

            int64_t edge_begin = xa(node);
            int64_t edge_end   = xa(node + 1);
            for (int64_t e = edge_begin; e < edge_end; e++) {
                int64_t neighbor = aj(e);
                EdgeWeight edge_weight = 1;

                PartitionID partTarget = stream_nodes_assign[neighbor];
                if (partSource == partTarget) {
                    blocks_sigma[partSource].first++;
                }
                blocks_sigma[partSource].second++;
            }
        }

        double modularity = onepass_partitioner->calculate_overall_score(
            blocks_weights, blocks_sigma, m);

        /* ── 6. Count clusters ── */
        int num_clusters = 0;
        for (auto& w : stream_blocks_weight) {
            if (w > 0) num_clusters++;
        }

        /* ── 7. Build result ── */
        py::array_t<int> result(n);
        auto r = result.mutable_unchecked<1>();
        for (int64_t i = 0; i < n; ++i) {
            r(i) = static_cast<int>(stream_nodes_assign[i]);
        }

        /* Cleanup */
        if (config.activeNodes_set != nullptr) {
            delete config.activeNodes_set;
            config.activeNodes_set = nullptr;
        }
        delete onepass_partitioner;

        restore_output();
        return py::make_tuple(num_clusters, modularity, result);

    } catch (...) {
        restore_output();
        if (config.activeNodes_set != nullptr) {
            delete config.activeNodes_set;
            config.activeNodes_set = nullptr;
        }
        delete onepass_partitioner;
        throw;
    }
}


PYBIND11_MODULE(_clustre, m) {
    m.doc() = "CluStRE streaming graph clustering";
    m.def("clustre_cluster", &py_clustre,
          py::arg("xadj"),
          py::arg("adjncy"),
          py::arg("mode") = "strong",
          py::arg("seed") = 0,
          py::arg("num_streams_passes") = 2,
          py::arg("resolution_param") = 0.5,
          py::arg("max_num_clusters") = -1,
          py::arg("ls_time_limit") = 600,
          py::arg("ls_frac_time") = 0.5,
          py::arg("cut_off") = 0.05,
          py::arg("suppress_output") = true,
          "Run CluStRE streaming graph clustering.\n\n"
          "Parameters:\n"
          "  xadj, adjncy   : CSR arrays (0-indexed)\n"
          "  mode            : clustering mode (light/light_plus/evo/strong)\n"
          "  seed            : random seed\n"
          "  num_streams_passes : number of streaming passes (default 2)\n"
          "  resolution_param : CPM resolution parameter (default 0.5)\n"
          "  max_num_clusters : maximum clusters (-1 = unlimited)\n"
          "  ls_time_limit   : local search time limit in seconds\n"
          "  ls_frac_time    : fraction of total time for local search\n"
          "  cut_off         : convergence cut-off for local search\n"
          "  suppress_output : suppress stdout/stderr\n\n"
          "Returns:\n"
          "  (num_clusters, modularity, assignment_array)\n"
    );
}
