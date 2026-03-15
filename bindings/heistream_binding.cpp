/**
 * heistream_binding.cpp — pybind11 binding for HeiStream streaming graph partitioner.
 *
 * Runs the full HeiStream algorithm (direct Fennel, BuffCut with priority buffer,
 * or no-priority buffer) by writing a temporary METIS file and driving the
 * HeiStream pipeline.  The partition result is extracted from the internal
 * StreamContext and returned as a numpy array.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* HeiStream headers */
#include "algorithms/bootstrap/register_builtin_algorithms.h"
#include "cli/configuration.h"
#include "core/context/stream_context.h"
#include "core/factory/algorithm_registry.h"
#include "core/interfaces/stream_algorithm.h"
#include "definitions.h"
#include "partition/partition_config.h"
#include "random_functions.h"

namespace py = pybind11;

namespace {

/* Write a METIS graph file from CSR arrays (0-indexed).
 * HeiStream expects 1-indexed METIS format. */
std::string write_temp_metis(
    const py::array_t<int64_t>& xadj,
    const py::array_t<int64_t>& adjncy)
{
    auto xa = xadj.unchecked<1>();
    auto aj = adjncy.unchecked<1>();
    int64_t n = xa.shape(0) - 1;
    int64_t m2 = aj.shape(0);           /* directed edges */
    int64_t m  = m2 / 2;                 /* undirected edges */

    char tmpname[] = "/tmp/heistream_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd < 0) throw std::runtime_error("Cannot create temp file for METIS graph");

    FILE* fp = fdopen(fd, "w");
    if (!fp) { close(fd); throw std::runtime_error("fdopen failed"); }

    fprintf(fp, "%lld %lld\n", (long long)n, (long long)m);
    for (int64_t v = 0; v < n; ++v) {
        if (xa(v) == xa(v + 1)) {
            /* Isolated node: write a non-empty line so HeiStream's
             * stream reader (which skips empty lines) doesn't shift
             * subsequent node indices. A lone space produces zero
             * parsed tokens — correct empty adjacency list. */
            fputc(' ', fp);
        }
        for (int64_t e = xa(v); e < xa(v + 1); ++e) {
            if (e > xa(v)) fputc(' ', fp);
            fprintf(fp, "%lld", (long long)(aj(e) + 1));   /* 1-indexed */
        }
        fputc('\n', fp);
    }
    fclose(fp);
    return std::string(tmpname);
}

} // namespace


static py::array_t<int> py_heistream(
    const py::array_t<int64_t>& xadj,
    const py::array_t<int64_t>& adjncy,
    int k,
    double imbalance,              /* percentage, e.g. 3 for 3% */
    int seed,
    int max_buffer_size,           /* 0 = auto/default from mode resolver */
    int batch_size,                /* MLP batch size */
    int num_streams_passes,        /* restreaming passes */
    bool run_parallel,
    bool suppress_output)
{
    /* 1. Write temp METIS file */
    std::string graph_file = write_temp_metis(xadj, adjncy);

    /* 2. Configure HeiStream — replicate the binary's init chain:
     *    standard() → strong() → stream_partition() → user overrides */
    Config config;
    {
        configuration cfg;
        cfg.standard(config);
        cfg.strong(config);
        cfg.stream_partition(config);      /* sets KaHIP + Fennel parameters */
    }
    config.k                  = k;
    config.imbalance          = imbalance;
    config.seed               = seed;
    config.graph_filename     = graph_file;
    config.suppress_output    = true;       /* never write partition file */
    config.run_parallel       = run_parallel;
    config.num_streams_passes = num_streams_passes;

    if (max_buffer_size > 0)  config.max_buffer_size = max_buffer_size;
    if (batch_size > 0)       config.batch_size      = batch_size;

    config.stream_global_epsilon = config.imbalance / 100.0;

    srand(config.seed);
    random_functions::setSeed(config.seed);

    /* 3. Register algorithms and run */
    algorithms::bootstrap::register_builtin_algorithms();

    /* Optionally suppress stdout/stderr from the C++ algorithm */
    std::streambuf* saved_cout = nullptr;
    std::streambuf* saved_cerr = nullptr;
    std::ofstream devnull;
    if (suppress_output) {
        devnull.open("/dev/null");
        saved_cout = std::cout.rdbuf(devnull.rdbuf());
        saved_cerr = std::cerr.rdbuf(devnull.rdbuf());
    }

    /* Replicate run_stream_engine but keep access to StreamContext */
    StreamContext ctx(config, StreamMode::Node);

    std::unique_ptr<IStreamAlgorithm> algorithm =
        core::factory::create_registered_algorithm(StreamMode::Node);
    if (!algorithm) {
        if (suppress_output) {
            std::cout.rdbuf(saved_cout);
            std::cerr.rdbuf(saved_cerr);
        }
        std::remove(graph_file.c_str());
        throw std::runtime_error("HeiStream: failed to create algorithm");
    }

    algorithm->prepare_run(config, graph_file, ctx);
    try {
        algorithm->run(config, graph_file, ctx);
    } catch (...) {
        if (suppress_output) {
            std::cout.rdbuf(saved_cout);
            std::cerr.rdbuf(saved_cerr);
        }
        std::remove(graph_file.c_str());
        throw;
    }

    /* 4. Extract partition BEFORE finalize_run (which frees the vectors) */
    if (!ctx.node_pass_state.stream_nodes_assign) {
        if (suppress_output) {
            std::cout.rdbuf(saved_cout);
            std::cerr.rdbuf(saved_cerr);
        }
        std::remove(graph_file.c_str());
        throw std::runtime_error("HeiStream: stream_nodes_assign is null after run");
    }
    const auto& assign = *ctx.node_pass_state.stream_nodes_assign;
    int64_t n = static_cast<int64_t>(assign.size());

    py::array_t<int> result(n);
    auto r = result.mutable_unchecked<1>();
    for (int64_t i = 0; i < n; ++i) {
        r(i) = static_cast<int>(assign[i]);
    }

    algorithm->finalize_run(config, graph_file, ctx);

    if (suppress_output) {
        std::cout.rdbuf(saved_cout);
        std::cerr.rdbuf(saved_cerr);
    }

    /* 5. Clean up temp file */
    std::remove(graph_file.c_str());

    return result;
}


PYBIND11_MODULE(_heistream, m) {
    m.doc() = "HeiStream streaming graph partitioner";
    m.def("heistream_partition", &py_heistream,
          py::arg("xadj"),
          py::arg("adjncy"),
          py::arg("k") = 2,
          py::arg("imbalance") = 3.0,
          py::arg("seed") = 0,
          py::arg("max_buffer_size") = 0,
          py::arg("batch_size") = 0,
          py::arg("num_streams_passes") = 1,
          py::arg("run_parallel") = false,
          py::arg("suppress_output") = true,
          "Run HeiStream streaming graph partitioner.\n\n"
          "Parameters:\n"
          "  xadj, adjncy : CSR arrays (0-indexed)\n"
          "  k             : number of partitions\n"
          "  imbalance     : allowed imbalance in percent (e.g. 3.0)\n"
          "  seed          : random seed\n"
          "  max_buffer_size : buffer size for BuffCut (0 = auto)\n"
          "  batch_size      : MLP batch size (0 = default)\n"
          "  num_streams_passes : number of streaming passes\n"
          "  run_parallel  : run parallel pipeline\n"
          "  suppress_output : suppress stdout/stderr\n"
    );
}
