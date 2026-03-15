#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <stdexcept>

#include "include/libsharedmap.h"
#include "include/libsharedmaptypes.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// String → enum helpers
// ---------------------------------------------------------------------------
static shared_map_strategy_type_t parse_strategy(const std::string &s) {
    if (s == "naive")    return NAIVE;
    if (s == "layer")    return LAYER;
    if (s == "queue")    return QUEUE;
    if (s == "nb_layer") return NB_LAYER;
    throw std::invalid_argument("Unknown strategy: '" + s +
        "'. Valid: naive, layer, queue, nb_layer");
}

static shared_map_algorithm_type_t parse_algorithm(const std::string &s) {
    if (s == "kaffpa_fast")               return KAFFPA_FAST;
    if (s == "kaffpa_eco")                return KAFFPA_ECO;
    if (s == "kaffpa_strong")             return KAFFPA_STRONG;
    if (s == "mtkahypar_default")         return MTKAHYPAR_DEFAULT;
    if (s == "mtkahypar_quality")         return MTKAHYPAR_QUALITY;
    if (s == "mtkahypar_highest_quality") return MTKAHYPAR_HIGHEST_QUALITY;
    throw std::invalid_argument("Unknown algorithm: '" + s +
        "'. Valid: kaffpa_fast, kaffpa_eco, kaffpa_strong, "
        "mtkahypar_default, mtkahypar_quality, mtkahypar_highest_quality");
}

// ---------------------------------------------------------------------------
// shared_map wrapper
// ---------------------------------------------------------------------------
static std::tuple<int, py::array_t<int>>
py_shared_map(py::array_t<int, py::array::c_style> vwgt,
              py::array_t<int, py::array::c_style> xadj,
              py::array_t<int, py::array::c_style> adjwgt,
              py::array_t<int, py::array::c_style> adjncy,
              py::array_t<int, py::array::c_style> hierarchy,
              py::array_t<int, py::array::c_style> distance,
              float imbalance,
              int n_threads,
              int seed,
              const std::string &strategy_str,
              const std::string &parallel_alg_str,
              const std::string &serial_alg_str,
              bool verbose)
{
    int n = static_cast<int>(xadj.size() - 1);
    int l = static_cast<int>(hierarchy.size());

    auto strategy     = parse_strategy(strategy_str);
    auto parallel_alg = parse_algorithm(parallel_alg_str);
    auto serial_alg   = parse_algorithm(serial_alg_str);

    py::array_t<int> partition(n);
    int comm_cost = 0;

    {
        py::gil_scoped_release release;
        shared_map_hierarchical_multisection(
            n,
            vwgt.mutable_data(),
            xadj.mutable_data(),
            adjwgt.mutable_data(),
            adjncy.mutable_data(),
            hierarchy.mutable_data(),
            distance.mutable_data(),
            l,
            imbalance,
            n_threads,
            seed,
            strategy,
            parallel_alg,
            serial_alg,
            comm_cost,
            partition.mutable_data(),
            verbose);
    }

    return std::make_tuple(comm_cost, partition);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_sharedmap, m) {
    m.doc() = "Python bindings for SharedMap hierarchical process mapping";

    m.def("shared_map", &py_shared_map,
          py::arg("vwgt"), py::arg("xadj"), py::arg("adjwgt"),
          py::arg("adjncy"), py::arg("hierarchy"), py::arg("distance"),
          py::arg("imbalance"), py::arg("n_threads"), py::arg("seed"),
          py::arg("strategy"), py::arg("parallel_alg"), py::arg("serial_alg"),
          py::arg("verbose"),
          R"doc(
          Run SharedMap hierarchical process mapping.

          Parameters
          ----------
          vwgt : ndarray[int32]
              Node weights (each >= 1).
          xadj : ndarray[int32]
              CSR row pointers (length n+1).
          adjwgt : ndarray[int32]
              Edge weights (each >= 1).
          adjncy : ndarray[int32]
              CSR column indices.
          hierarchy : ndarray[int32]
              Hierarchy levels (e.g. [4, 8] for 4 nodes x 8 cores).
          distance : ndarray[int32]
              Communication distances per hierarchy level.
          imbalance : float
              Allowed imbalance (e.g. 0.03 = 3%).
          n_threads : int
              Number of threads.
          seed : int
              Random seed.
          strategy : str
              Thread distribution strategy.
          parallel_alg : str
              Parallel partitioning algorithm.
          serial_alg : str
              Serial partitioning algorithm.
          verbose : bool
              Print statistics.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (communication_cost, assignment).
          )doc");
}
