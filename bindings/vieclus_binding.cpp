#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "vieclus_interface.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// vieclus_clustering wrapper
// ---------------------------------------------------------------------------
static std::tuple<double, int, py::array_t<int>>
py_cluster(py::array_t<int, py::array::c_style> vwgt,
           py::array_t<int, py::array::c_style> xadj,
           py::array_t<int, py::array::c_style> adjcwgt,
           py::array_t<int, py::array::c_style> adjncy,
           bool suppress_output, int seed,
           double time_limit, int cluster_upperbound)
{
    int n = static_cast<int>(xadj.size() - 1);

    int* vwgt_ptr    = vwgt.size()   > 0 ? vwgt.mutable_data()   : nullptr;
    int* adjcwgt_ptr = adjcwgt.size() > 0 ? adjcwgt.mutable_data() : nullptr;

    py::array_t<int> clustering(n);
    double modularity = 0.0;
    int num_clusters = 0;

    vieclus_clustering(&n, vwgt_ptr, xadj.mutable_data(),
                       adjcwgt_ptr, adjncy.mutable_data(),
                       suppress_output, seed,
                       time_limit, cluster_upperbound,
                       &modularity, &num_clusters, clustering.mutable_data());

    return std::make_tuple(modularity, num_clusters, clustering);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_vieclus, m) {
    m.doc() = "Python bindings for VieClus graph clustering";

    m.def("cluster", &py_cluster,
          py::arg("vwgt"), py::arg("xadj"), py::arg("adjcwgt"),
          py::arg("adjncy"), py::arg("suppress_output"), py::arg("seed"),
          py::arg("time_limit"), py::arg("cluster_upperbound"),
          R"doc(
          Run VieClus graph clustering.

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
          suppress_output : bool
              Suppress VieClus console output.
          seed : int
              Random seed.
          time_limit : float
              Time limit in seconds.
          cluster_upperbound : int
              Maximum cluster size (0 = no limit).

          Returns
          -------
          tuple[float, int, ndarray[int32]]
              (modularity, num_clusters, cluster_assignment).
          )doc");
}
