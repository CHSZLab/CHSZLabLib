#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>

// KaLP headers — note: "configuration.h" resolves to KaHIP's version
// via include path order, so we include KaLP's configuration explicitly.
#include "undirected_graph.h"
#include "config.h"
#include "partitioning.h"
#include "QueryManager.hpp"

namespace py = pybind11;

static std::tuple<int, py::array_t<int>>
py_longest_path(py::array_t<int, py::array::c_style> xadj,
                py::array_t<int, py::array::c_style> adjncy,
                py::array_t<int, py::array::c_style> ewgt,
                int start_vertex,
                int target_vertex,
                std::string partition_config_str,
                int block_size,
                int number_of_threads,
                int split_steps,
                int threshold) {

    int n = static_cast<int>(xadj.size() - 1);
    int *xadj_ptr = xadj.mutable_data();
    int *adjncy_ptr = adjncy.mutable_data();
    int *ewgt_ptr = ewgt.size() > 0 ? ewgt.mutable_data() : nullptr;

    // Build KaLP's UndirectedGraph from CSR
    UndirectedGraph G;
    for (int i = 0; i < n; i++) {
        G.addNode();
    }
    for (int u = 0; u < n; u++) {
        for (int idx = xadj_ptr[u]; idx < xadj_ptr[u + 1]; idx++) {
            int v = adjncy_ptr[idx];
            if (u < v) {
                int w = ewgt_ptr ? ewgt_ptr[idx] : 1;
                G.addEdge(u, v, w);
            }
        }
    }

    // Set up Config with KaLP defaults (inlined from configuration_lp::standard)
    Config config;
    config.seed = 0;
    config.start_vertex = UNDEFINED_NODE;
    config.target_vertex = UNDEFINED_NODE;
    config.print_path = false;
    config.number_of_blocks = 1;
    config.block_size = 10;
    config.output_filename = "";
    config.partition_configuration = PARTITION_CONFIG_ECO;
    config.subgraph_size = 100;
    config.number_of_threads = 1;
    config.split_steps = 0;
    config.threshold = 0;

    config.start_vertex = start_vertex;
    config.target_vertex = target_vertex;
    config.block_size = block_size;
    config.number_of_threads = static_cast<unsigned>(number_of_threads);
    config.split_steps = static_cast<unsigned>(split_steps);
    config.threshold = static_cast<unsigned>(threshold);

    if (partition_config_str == "strong") {
        config.partition_configuration = PARTITION_CONFIG_STRONG;
    } else if (partition_config_str == "fast") {
        config.partition_configuration = PARTITION_CONFIG_FAST;
    } else {
        config.partition_configuration = PARTITION_CONFIG_ECO;
    }

    int number_of_blocks = G.size() / block_size;
    if (G.size() % block_size) {
        number_of_blocks++;
    }

    // Suppress stdout/stderr during algorithm run
    std::streambuf *old_cout = std::cout.rdbuf();
    std::streambuf *old_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());
    std::cerr.rdbuf(null_stream.rdbuf());

    std::vector<std::vector<int>> partitions;
    partitionGraph(&G, config, &partitions, number_of_blocks, 2);

    QueryManager q(&G, partitions, start_vertex, target_vertex);
    Result result = q.run(config.number_of_threads, config.split_steps, config.threshold);

    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);

    int length = result.weight;
    int path_size = static_cast<int>(result.path.size());
    py::array_t<int> path_arr(path_size);
    auto r = path_arr.mutable_unchecked<1>();
    for (int i = 0; i < path_size; i++) {
        r(i) = static_cast<int>(result.path[i]);
    }

    return std::make_tuple(length, path_arr);
}

PYBIND11_MODULE(_kalp, m) {
    m.doc() = "Python bindings for KaLP (Karlsruhe Longest Paths)";

    m.def("longest_path", &py_longest_path,
          py::arg("xadj"), py::arg("adjncy"), py::arg("ewgt"),
          py::arg("start_vertex"), py::arg("target_vertex"),
          py::arg("partition_config"), py::arg("block_size"),
          py::arg("number_of_threads"), py::arg("split_steps"),
          py::arg("threshold"),
          R"doc(
          Compute longest simple path between start and target vertex.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (path_length, path_vertices). path_length is 0 if no path exists.
          )doc");
}
