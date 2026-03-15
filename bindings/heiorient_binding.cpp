#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ds/empty_problem.h"
#include "edge_orientation/ds/smaller_graph.h"
#include "edge_orientation/tools/combined.h"
#include "edge_orientation/tools/dfs.h"

namespace py = pybind11;

using SimpleSetGraph = rpo::ds::SimpleSetGraph<uint32_t>;
using SmallerGraph = rpo::edge_orientation::ds::SmallerGraph<unsigned int, size_t>;

static std::tuple<int, py::array_t<int>, py::array_t<int>>
py_orient_edges(
        py::array_t<int, py::array::c_style> xadj,
        py::array_t<int, py::array::c_style> adjncy,
        const std::string& algorithm,
        int seed,
        int eager_size) {

    int n = static_cast<int>(xadj.size() - 1);
    const int *xadj_ptr = xadj.data();
    const int *adjncy_ptr = adjncy.data();

    // Count undirected edges (each undirected edge appears twice in CSR)
    int m_directed = static_cast<int>(adjncy.size());
    int m = m_directed / 2;

    if (n == 0 || m == 0) {
        py::array_t<int> out_degrees(n);
        py::array_t<int> edge_heads(m_directed);
        auto od = out_degrees.mutable_unchecked<1>();
        auto eh = edge_heads.mutable_unchecked<1>();
        for (int i = 0; i < n; i++) od(i) = 0;
        for (int i = 0; i < m_directed; i++) eh(i) = 0;
        return std::make_tuple(0, out_degrees, edge_heads);
    }

    // Build SimpleSetGraph from CSR
    SimpleSetGraph ssg(m, n);
    for (int u = 0; u < n; u++) {
        for (int idx = xadj_ptr[u]; idx < xadj_ptr[u + 1]; idx++) {
            int v = adjncy_ptr[idx];
            if (u < v) {
                ssg.addEdge(static_cast<uint32_t>(u),
                            static_cast<uint32_t>(v), 1);
            }
        }
    }
    ssg.finish();

    // Convert to SmallerGraph (flat-array structure for DFS/combined)
    SmallerGraph sg(ssg);

    // Initial balanced orientation
    sg.resort_fast();

    // Dispatch algorithm
    if (algorithm == "two_approx") {
        // resort_fast() already gives a 2-approximation; nothing more
    } else if (algorithm == "dfs") {
        rpo::edge_orientation::dfs::solve_by_dfs(sg);
    } else if (algorithm == "combined") {
        auto max_flow = sg.computeOutflows();
        int lower_bound = static_cast<int>(
            std::ceil(static_cast<double>(sg.edge_count()) /
                      static_cast<double>(sg.vertex_count())));
        int count = std::max(1, static_cast<int>(
            std::sqrt(static_cast<double>(max_flow - lower_bound))));
        rpo::edge_orientation::combined::solve_by_dfs_combined_multiple(
            sg, count, true, static_cast<size_t>(eager_size));
    } else {
        throw py::value_error(
            "Unknown algorithm '" + algorithm + "'. "
            "Choose from: 'two_approx', 'dfs', 'combined'.");
    }

    // Extract results
    int max_out_degree = static_cast<int>(sg.computeOutflows());

    py::array_t<int> out_degrees(n);
    auto od = out_degrees.mutable_unchecked<1>();
    for (int i = 0; i < n; i++) {
        od(i) = static_cast<int>(sg._nodes_out[i]);
    }

    // Build edge_heads: for each CSR entry (u, adjncy[idx]),
    // edge_heads[idx] = 1 if oriented u->v, 0 if oriented v->u.
    //
    // In SmallerGraph, out-neighbors of u are stored at:
    //   flat_edge_info[_nodes_offset[u] .. _nodes_offset[u] + _nodes_out[u])
    // Build a set of out-neighbors per node, then scan CSR.

    py::array_t<int> edge_heads(m_directed);
    auto eh = edge_heads.mutable_unchecked<1>();

    // For each node u, collect its out-neighbors into a temporary structure
    // We use a boolean marker array (size n) to avoid O(n*degree) per node
    std::vector<bool> is_out_neighbor(n, false);

    for (int u = 0; u < n; u++) {
        // Mark out-neighbors
        size_t offset = sg._nodes_offset[u];
        int out_count = sg._nodes_out[u];
        for (int j = 0; j < out_count; j++) {
            int v = static_cast<int>(sg.flat_edge_info[offset + j]);
            is_out_neighbor[v] = true;
        }

        // Scan CSR entries for u
        for (int idx = xadj_ptr[u]; idx < xadj_ptr[u + 1]; idx++) {
            int v = adjncy_ptr[idx];
            eh(idx) = is_out_neighbor[v] ? 1 : 0;
        }

        // Clear marks
        for (int j = 0; j < out_count; j++) {
            int v = static_cast<int>(sg.flat_edge_info[offset + j]);
            is_out_neighbor[v] = false;
        }
    }

    return std::make_tuple(max_out_degree, out_degrees, edge_heads);
}

PYBIND11_MODULE(_heiorient, m) {
    m.doc() = "Python bindings for HeiOrient edge orientation algorithms";

    m.def("orient_edges", &py_orient_edges,
          py::arg("xadj"), py::arg("adjncy"),
          py::arg("algorithm"), py::arg("seed"),
          py::arg("eager_size"),
          R"doc(
          Orient edges of an undirected graph to minimize maximum out-degree.

          Parameters
          ----------
          xadj : ndarray[int32]
              CSR row pointers (length n+1).
          adjncy : ndarray[int32]
              CSR column indices (length 2*m).
          algorithm : str
              Algorithm name: 'two_approx', 'dfs', or 'combined'.
          seed : int
              Random seed (unused by current algorithms, reserved).
          eager_size : int
              Eager threshold for the combined algorithm.

          Returns
          -------
          tuple[int, ndarray[int32], ndarray[int32]]
              (max_out_degree, out_degrees, edge_heads).
          )doc");
}
