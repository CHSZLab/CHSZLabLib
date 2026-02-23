#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdlib>

#include "kaHIP_interface.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// kaffpa wrapper
// ---------------------------------------------------------------------------
static std::tuple<int, py::array_t<int>>
py_kaffpa(py::array_t<int, py::array::c_style> vwgt,
          py::array_t<int, py::array::c_style> xadj,
          py::array_t<int, py::array::c_style> adjcwgt,
          py::array_t<int, py::array::c_style> adjncy,
          int nparts, double imbalance, bool suppress_output,
          int seed, int mode)
{
    int n = static_cast<int>(xadj.size() - 1);

    int* vwgt_ptr   = vwgt.size()   > 0 ? vwgt.mutable_data()   : nullptr;
    int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

    py::array_t<int> part(n);
    int edgecut = 0;

    kaffpa(&n, vwgt_ptr, xadj.mutable_data(),
           adjcwgt_ptr, adjncy.mutable_data(),
           &nparts, &imbalance, suppress_output, seed, mode,
           &edgecut, part.mutable_data());

    return std::make_tuple(edgecut, part);
}

// ---------------------------------------------------------------------------
// node_separator wrapper
// ---------------------------------------------------------------------------
static std::tuple<int, py::array_t<int>>
py_node_separator(py::array_t<int, py::array::c_style> vwgt,
                  py::array_t<int, py::array::c_style> xadj,
                  py::array_t<int, py::array::c_style> adjcwgt,
                  py::array_t<int, py::array::c_style> adjncy,
                  int nparts, double imbalance, bool suppress_output,
                  int seed, int mode)
{
    int n = static_cast<int>(xadj.size() - 1);

    int* vwgt_ptr    = vwgt.size()   > 0 ? vwgt.mutable_data()    : nullptr;
    int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

    int  num_separator_vertices = 0;
    int* separator_raw          = nullptr;

    node_separator(&n, vwgt_ptr, xadj.mutable_data(),
                   adjcwgt_ptr, adjncy.mutable_data(),
                   &nparts, &imbalance, suppress_output, seed, mode,
                   &num_separator_vertices, &separator_raw);

    // Copy the C-allocated array into a numpy array, then free.
    py::array_t<int> separator(num_separator_vertices);
    if (num_separator_vertices > 0 && separator_raw != nullptr) {
        std::memcpy(separator.mutable_data(), separator_raw,
                    num_separator_vertices * sizeof(int));
        std::free(separator_raw);
    }

    return std::make_tuple(num_separator_vertices, separator);
}

// ---------------------------------------------------------------------------
// reduced_nd (node ordering) wrapper
// ---------------------------------------------------------------------------
static py::array_t<int>
py_node_ordering(py::array_t<int, py::array::c_style> xadj,
                 py::array_t<int, py::array::c_style> adjncy,
                 bool suppress_output, int seed, int mode)
{
    int n = static_cast<int>(xadj.size() - 1);

    py::array_t<int> ordering(n);

    reduced_nd(&n, xadj.mutable_data(), adjncy.mutable_data(),
               suppress_output, seed, mode,
               ordering.mutable_data());

    return ordering;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_kahip, m) {
    m.doc() = "Python bindings for KaHIP graph partitioning";

    // Mode constants
    m.attr("FAST")        = 0;
    m.attr("ECO")         = 1;
    m.attr("STRONG")      = 2;
    m.attr("FASTSOCIAL")  = 3;
    m.attr("ECOSOCIAL")   = 4;
    m.attr("STRONGSOCIAL") = 5;

    m.def("kaffpa", &py_kaffpa,
          py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
          py::arg("adjncy"), py::arg("nparts"), py::arg("imbalance"),
          py::arg("suppress_output"), py::arg("seed"), py::arg("mode"),
          R"doc(
          Run KaHIP's kaffpa graph partitioner.

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
          seed : int
              Random seed.
          mode : int
              Quality mode (FAST=0, ECO=1, STRONG=2, ...).

          Returns
          -------
          tuple[int, ndarray[int32]]
              (edgecut, partition_assignment).
          )doc");

    m.def("node_separator", &py_node_separator,
          py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
          py::arg("adjncy"), py::arg("nparts"), py::arg("imbalance"),
          py::arg("suppress_output"), py::arg("seed"), py::arg("mode"),
          R"doc(
          Compute a node separator using KaHIP.

          Returns
          -------
          tuple[int, ndarray[int32]]
              (num_separator_vertices, separator_node_ids).
          )doc");

    m.def("node_ordering", &py_node_ordering,
          py::arg("xadj"), py::arg("adjncy"),
          py::arg("suppress_output"), py::arg("seed"), py::arg("mode"),
          R"doc(
          Compute a reduced nested dissection ordering using KaHIP.

          Returns
          -------
          ndarray[int32]
              Permutation array of length n.
          )doc");
}
