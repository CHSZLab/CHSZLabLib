#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "kaHIP_interface.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// kaffpaE wrapper
// ---------------------------------------------------------------------------
static std::tuple<int, double, py::array_t<int>>
py_kaffpaE(py::array_t<int, py::array::c_style> vwgt,
           py::array_t<int, py::array::c_style> xadj,
           py::array_t<int, py::array::c_style> adjcwgt,
           py::array_t<int, py::array::c_style> adjncy,
           int nparts, double imbalance, bool suppress_output,
           bool graph_partitioned,
           py::array_t<int, py::array::c_style> initial_partition,
           int time_limit, int seed, int mode)
{
    int n = static_cast<int>(xadj.size() - 1);

    int* vwgt_ptr    = vwgt.size()    > 0 ? vwgt.mutable_data()    : nullptr;
    int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

    py::array_t<int> part(n);
    int edgecut  = 0;
    double balance = 0.0;

    // If warm-starting, copy the initial partition into the output array
    if (graph_partitioned && initial_partition.size() == n) {
        std::memcpy(part.mutable_data(), initial_partition.data(),
                    n * sizeof(int));
    }

    kaffpaE(&n, vwgt_ptr, xadj.mutable_data(),
            adjcwgt_ptr, adjncy.mutable_data(),
            &nparts, &imbalance, suppress_output,
            graph_partitioned, time_limit, seed, mode,
            MPI_COMM_WORLD,
            &edgecut, &balance, part.mutable_data());

    return std::make_tuple(edgecut, balance, part);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_kahipe, m) {
    m.doc() = "Python bindings for KaHIP evolutionary graph partitioning (kaffpaE)";

    // Mode constants
    m.attr("FAST")           = 0;
    m.attr("ECO")            = 1;
    m.attr("STRONG")         = 2;
    m.attr("FASTSOCIAL")     = 3;
    m.attr("ECOSOCIAL")      = 4;
    m.attr("STRONGSOCIAL")   = 5;
    m.attr("ULTRAFASTSOCIAL") = 6;

    m.def("kaffpaE", &py_kaffpaE,
          py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
          py::arg("adjncy"), py::arg("nparts"), py::arg("imbalance"),
          py::arg("suppress_output"), py::arg("graph_partitioned"),
          py::arg("initial_partition"),
          py::arg("time_limit"), py::arg("seed"), py::arg("mode"),
          R"doc(
          Run KaHIP's evolutionary/memetic graph partitioner (kaffpaE).

          Parameters
          ----------
          vwgt : ndarray[int32]
              Node weights (empty array for unit weights).
          xadj : ndarray[int32]
              CSR row pointers.
          adjcwgt : ndarray[int32]
              Edge weights (empty array for unit weights).
          adjncy : ndarray[int32]
              CSR column indices.
          nparts : int
              Number of partitions.
          imbalance : float
              Allowed imbalance (e.g. 0.03 = 3%).
          suppress_output : bool
              Suppress KaHIP console output.
          graph_partitioned : bool
              Whether an initial partition is provided.
          initial_partition : ndarray[int32]
              Initial partition (used when graph_partitioned is True).
          time_limit : int
              Time limit in seconds for the evolutionary algorithm.
          seed : int
              Random seed.
          mode : int
              Quality mode (FAST=0, ECO=1, STRONG=2, ...).

          Returns
          -------
          tuple[int, float, ndarray[int32]]
              (edgecut, balance, partition_assignment).
          )doc");
}
