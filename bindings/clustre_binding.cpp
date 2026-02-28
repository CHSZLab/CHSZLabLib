/**
 * clustre_binding.cpp — pybind11 binding for CluStRE streaming graph clustering.
 *
 * Runs the CluStRE streaming clustering algorithm by writing a temporary METIS
 * file and driving the streaming pipeline.  The clustering result (assignment,
 * modularity, num_clusters) is returned to Python.
 *
 * Follows the same pattern as heistream_binding.cpp.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
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

namespace py = pybind11;

namespace {

/**
 * Read METIS header and initialize the streaming config.
 * Sets total_nodes, total_edges, n_batches, and opens the stream.
 */
void init_stream_config(HeiClus::PartitionConfig& config,
                        const std::string& graph_file,
                        std::ifstream& stream_in) {
    stream_in.open(graph_file);
    if (!stream_in.good()) {
        throw std::runtime_error("CluStRE: cannot open graph file: " + graph_file);
    }

    /* Read the header line: "n m [fmt]" */
    std::string line;
    while (std::getline(stream_in, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }
    std::istringstream header(line);
    uint64_t n = 0, m = 0;
    header >> n >> m;

    config.total_nodes = n;
    config.total_edges = m;  /* undirected edges */
    config.n_batches = n;
    config.remaining_stream_nodes = n;
    config.stream_n_nodes = n;
    config.stream_in = &stream_in;
}

/**
 * Parse one adjacency line from the METIS file.
 * Returns the list of 1-indexed neighbor IDs (matching CluStRE convention).
 */
std::vector<std::vector<uint64_t>>* read_one_node(std::ifstream& stream_in) {
    auto* input = new std::vector<std::vector<uint64_t>>(1);
    std::string line;
    if (!std::getline(stream_in, line)) {
        (*input)[0].clear();
        return input;
    }
    /* Skip comment lines */
    while (!line.empty() && line[0] == '%') {
        if (!std::getline(stream_in, line)) {
            (*input)[0].clear();
            return input;
        }
    }
    std::istringstream ss(line);
    uint64_t val;
    while (ss >> val) {
        (*input)[0].push_back(val);  /* already 1-indexed in METIS format */
    }
    return input;
}

/**
 * Write a METIS graph file from CSR arrays (0-indexed).
 * CluStRE expects 1-indexed METIS format.
 */
std::string write_temp_metis(
    const py::array_t<int64_t>& xadj,
    const py::array_t<int64_t>& adjncy)
{
    auto xa = xadj.unchecked<1>();
    auto aj = adjncy.unchecked<1>();
    int64_t n = xa.shape(0) - 1;
    int64_t m2 = aj.shape(0);
    int64_t m  = m2 / 2;

    char tmpname[] = "/tmp/clustre_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd < 0) throw std::runtime_error("Cannot create temp file for METIS graph");

    FILE* fp = fdopen(fd, "w");
    if (!fp) { close(fd); throw std::runtime_error("fdopen failed"); }

    fprintf(fp, "%lld %lld\n", (long long)n, (long long)m);
    for (int64_t v = 0; v < n; ++v) {
        if (xa(v) == xa(v + 1)) {
            fputc(' ', fp);
        }
        for (int64_t e = xa(v); e < xa(v + 1); ++e) {
            if (e > xa(v)) fputc(' ', fp);
            fprintf(fp, "%lld", (long long)(aj(e) + 1));
        }
        fputc('\n', fp);
    }
    fclose(fp);
    return std::string(tmpname);
}

} // namespace


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
    /* 1. Write temp METIS file */
    std::string graph_file = write_temp_metis(xadj, adjncy);

    /* 2. Initialize CluStRE config */
    HeiClus::PartitionConfig config;
    {
        HeiClus::configuration cfg;
        cfg.standard(config);
        cfg.strong(config);
        cfg.stream_partition(config);
    }

    config.seed = seed;
    config.suppress_output = true;
    config.suppress_file_output = true;
    config.stream_input = true;
    config.rle_length = -1;   /* use std::vector, not compression */
    config.cpm_gamma = resolution_param;
    config.ls_time_limit = ls_time_limit;
    config.ls_frac_time = ls_frac_time;
    config.cut_off = cut_off;
    config.ext_clustering_algorithm = NO_EXT_ALGORITHM;

    /* Parse mode string */
    if (mode_str == "light") {
        config.mode = LIGHT;
        config.restream_amount = 0;
        config.one_pass_algorithm = ONEPASS_MODULARITY;
    } else if (mode_str == "light_plus") {
        config.mode = LIGHT_PLUS;
        config.one_pass_algorithm = ONEPASS_MODULARITY;
    } else if (mode_str == "evo") {
        config.mode = EVO;
        config.restream_amount = 0;
        config.one_pass_algorithm = ONEPASS_MODULARITY;
    } else if (mode_str == "strong") {
        config.mode = STRONG;
        config.one_pass_algorithm = ONEPASS_MODULARITY;
    } else {
        std::remove(graph_file.c_str());
        throw std::runtime_error("CluStRE: unknown mode '" + mode_str +
                                 "'. Choose from: light, light_plus, evo, strong");
    }

    if (max_num_clusters > 0) {
        config.max_num_clusters = max_num_clusters;
    }

    if (num_streams_passes > 1) {
        config.restream_amount = num_streams_passes - 1;
    }

    config.stream_buffer_len = 1;

    /* Optionally suppress stdout/stderr */
    std::streambuf* saved_cout = nullptr;
    std::streambuf* saved_cerr = nullptr;
    std::ofstream devnull;
    if (suppress_output) {
        devnull.open("/dev/null");
        saved_cout = std::cout.rdbuf(devnull.rdbuf());
        saved_cerr = std::cerr.rdbuf(devnull.rdbuf());
    }

    try {
        /* 3. Read header and init */
        std::ifstream stream_in;
        init_stream_config(config, graph_file, stream_in);

        uint64_t n = config.total_nodes;

        /* Allocate assignment vector */
        std::vector<PartitionID> stream_nodes_assign(n, INVALID_PARTITION);
        config.stream_nodes_assign = &stream_nodes_assign;

        std::vector<NodeWeight> stream_blocks_weight;
        config.stream_blocks_weight = &stream_blocks_weight;

        config.neighbor_blocks.resize(1);
        config.all_blocks_to_keys.resize(1);
        config.next_key.resize(1);
        config.next_key[0] = 0;

        if (max_num_clusters <= 0) {
            config.max_num_clusters = static_cast<int>(n);
        }

        /* Initialize partitioner */
        vertex_partitioning* onepass_partitioner = nullptr;
        onepass_partitioner = new onepass_modularity(
            0, 0, config.total_nodes, 1, config.mode, false, config.cpm_gamma);

        /* Partial offsets for local search (LIGHT_PLUS / STRONG modes) */
        std::vector<std::streampos> partial_offsets;
        if (config.mode == LIGHT_PLUS || config.mode == STRONG) {
            config.partialOffsets = &partial_offsets;
            config.activeNodes_set = new robin_hood::unordered_set<uint64_t>();
        }

        /* 4. Main streaming loop */
        uint64_t num_lines = 1;

        for (int restreaming = 0; restreaming < config.restream_amount + 1; restreaming++) {
            if (restreaming) {
                /* Reset file for restreaming */
                stream_in.clear();
                stream_in.seekg(0);
                /* Skip header */
                std::string hdr;
                while (std::getline(stream_in, hdr)) {
                    if (hdr.empty() || hdr[0] == '%') continue;
                    break;
                }
                config.remaining_stream_nodes = n;

                /* Reset blocks edge weights */
                while (onepass_partitioner->blocks.size() > stream_blocks_weight.size()) {
                    onepass_partitioner->blocks.pop_back();
                    config.clusters_to_ix_mapping.pop_back();
                    config.neighbor_blocks[0].pop_back();
                }
                for (size_t p = 0; p < onepass_partitioner->blocks.size(); p++) {
                    onepass_partitioner->blocks[p].e_weight = 0;
                }
                config.next_key[0] = 0;
            }

            for (uint64_t curr_node = 0; curr_node < n; curr_node++) {
                int my_thread = 0;

                /* Read one node from file */
                auto* input = read_one_node(stream_in);

                /* Store partial offsets for local search */
                if ((curr_node % config.offset_interval == 0 || curr_node == 0)
                    && restreaming == 1
                    && (config.mode == LIGHT_PLUS || config.mode == STRONG)) {
                    // We record stream positions for offset-based access
                    // but in our simplified binding, local search uses full re-read
                }

                /* Process node: build neighbor blocks */
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

                std::vector<uint64_t>& line_numbers = (*input)[0];

                for (size_t col = 0; col < line_numbers.size(); col++) {
                    uint64_t target = line_numbers[col];
                    EdgeWeight edge_weight = 1;
                    onepass_partitioner->increment_graph_edge_count(edge_weight, restreaming);

                    PartitionID targetPar = INVALID_PARTITION;
                    if (target >= 1 && (target - 1) < n) {
                        targetPar = stream_nodes_assign[target - 1];
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

                /* Update quotient graph for evo/strong modes */
                if (restreaming == 0 && config.mode != LIGHT && config.mode != LIGHT_PLUS) {
                    onepass_partitioner->update_quotient_graph(block, config.neighbor_blocks[my_thread]);
                }

                PartitionID orig_part = static_cast<PartitionID>(-1);
                if (restreaming) {
                    orig_part = stream_nodes_assign[curr_node];
                }
                stream_nodes_assign[curr_node] = block;

                /* Track active nodes for local search */
                if (restreaming == config.restream_amount
                    && orig_part != block
                    && (config.mode == LIGHT_PLUS || config.mode == STRONG)
                    && config.activeNodes_set != nullptr) {
                    for (auto& neighbour : line_numbers) {
                        if (neighbour >= 1) {
                            config.activeNodes_set->insert(neighbour - 1);
                        }
                    }
                }

                delete input;

                config.previous_assignment = block;
                if (stream_blocks_weight.size() <= static_cast<size_t>(block)) {
                    stream_blocks_weight.resize(block + 1, 0);
                }
                stream_blocks_weight[block] += 1;
                if (restreaming && orig_part != static_cast<PartitionID>(-1)) {
                    stream_blocks_weight[orig_part] -= 1;
                }
            }
        }

        /* 5. Local search phase (STRONG / LIGHT_PLUS) */
        if ((config.mode == STRONG || config.mode == LIGHT_PLUS)
            && config.activeNodes_set != nullptr
            && !config.activeNodes_set->empty()) {

            auto start_time = std::chrono::steady_clock::now();
            std::chrono::seconds TIME_LIMIT(config.ls_time_limit);
            bool active_nodes_exist = true;

            while (active_nodes_exist) {
                auto* new_active = new robin_hood::unordered_set<uint64_t>();
                onepass_partitioner->curr_round_delta_mod = 0;
                active_nodes_exist = false;

                for (size_t p = 0; p < onepass_partitioner->blocks.size(); p++) {
                    onepass_partitioner->blocks[p].e_weight = 0;
                }
                config.next_key[0] = 0;
                bool timeStop = false;

                for (const auto& curr_node : *config.activeNodes_set) {
                    /* Re-read the node's neighbors from file using seekg */
                    stream_in.clear();
                    stream_in.seekg(0);
                    /* Skip header */
                    std::string hdr;
                    while (std::getline(stream_in, hdr)) {
                        if (hdr.empty() || hdr[0] == '%') continue;
                        break;
                    }
                    /* Skip to the right line */
                    for (uint64_t skip = 0; skip < curr_node; skip++) {
                        std::string tmp;
                        std::getline(stream_in, tmp);
                    }
                    auto* input = read_one_node(stream_in);

                    /* Process node */
                    int my_thread = 0;
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

                    std::vector<uint64_t>& line_numbers = (*input)[0];
                    for (size_t col = 0; col < line_numbers.size(); col++) {
                        uint64_t target = line_numbers[col];
                        EdgeWeight edge_weight = 1;
                        onepass_partitioner->increment_graph_edge_count(edge_weight, 1);

                        PartitionID targetPar = INVALID_PARTITION;
                        if (target >= 1 && (target - 1) < n) {
                            targetPar = stream_nodes_assign[target - 1];
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
                        for (auto& neighbour : line_numbers) {
                            if (neighbour >= 1) {
                                new_active->insert(neighbour - 1);
                            }
                        }
                    }

                    delete input;
                    config.previous_assignment = block;
                    if (stream_blocks_weight.size() <= static_cast<size_t>(block)) {
                        stream_blocks_weight.resize(block + 1, 0);
                    }
                    stream_blocks_weight[block] += 1;
                    stream_blocks_weight[orig_part] -= 1;

                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    if (elapsed > TIME_LIMIT) {
                        timeStop = true;
                        break;
                    }
                }

                delete config.activeNodes_set;
                config.activeNodes_set = new_active;

                if (onepass_partitioner->curr_round_delta_mod <
                    onepass_partitioner->overall_delta_mod * config.cut_off) {
                    break;
                }
                if (timeStop) {
                    active_nodes_exist = false;
                }
                onepass_partitioner->overall_delta_mod += onepass_partitioner->curr_round_delta_mod;
            }
        }

        /* 6. Count clusters and compute approximate modularity */
        int num_clusters = 0;
        for (auto& w : stream_blocks_weight) {
            if (w > 0) num_clusters++;
        }

        double modularity = onepass_partitioner->overall_delta_mod;

        /* 7. Extract result */
        py::array_t<int> result(n);
        auto r = result.mutable_unchecked<1>();
        for (uint64_t i = 0; i < n; ++i) {
            r(i) = static_cast<int>(stream_nodes_assign[i]);
        }

        /* Cleanup */
        if (config.activeNodes_set != nullptr) {
            delete config.activeNodes_set;
            config.activeNodes_set = nullptr;
        }
        delete onepass_partitioner;
        stream_in.close();

        if (suppress_output) {
            std::cout.rdbuf(saved_cout);
            std::cerr.rdbuf(saved_cerr);
        }
        std::remove(graph_file.c_str());

        return py::make_tuple(num_clusters, modularity, result);

    } catch (...) {
        if (suppress_output) {
            std::cout.rdbuf(saved_cout);
            std::cerr.rdbuf(saved_cerr);
        }
        std::remove(graph_file.c_str());
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
