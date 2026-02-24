#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "mc-graph.hpp"

namespace py = pybind11;

struct OutputSuppressor {
    std::streambuf *old_cout, *old_cerr;
    std::ostringstream sink_out, sink_err;
    OutputSuppressor()
        : old_cout(std::cout.rdbuf(sink_out.rdbuf())),
          old_cerr(std::cerr.rdbuf(sink_err.rdbuf())) {}
    ~OutputSuppressor() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
};

static MaxCutGraph build_graph(
        py::array_t<int64_t, py::array::c_style> xadj,
        py::array_t<int32_t, py::array::c_style> adjncy,
        py::array_t<int64_t, py::array::c_style> adjwgt) {

    int n = static_cast<int>(xadj.size() - 1);
    const int64_t *xadj_ptr = xadj.data();
    const int32_t *adjncy_ptr = adjncy.data();
    const int64_t *adjwgt_ptr = adjwgt.data();
    bool has_weights = (adjwgt.size() > 0);

    std::vector<std::tuple<int, int, EdgeWeight>> elist;
    for (int u = 0; u < n; u++) {
        for (int64_t idx = xadj_ptr[u]; idx < xadj_ptr[u + 1]; idx++) {
            int v = adjncy_ptr[idx];
            if (u < v) {
                EdgeWeight w = has_weights ? static_cast<EdgeWeight>(adjwgt_ptr[idx]) : 1;
                elist.emplace_back(u, v, w);
            }
        }
    }

    return MaxCutGraph(elist, n);
}

// Kernelize + heuristic solve.
// Returns (cut_value, partition_array).
static std::tuple<long long, py::array_t<int32_t>>
py_maxcut_heuristic(
        py::array_t<int64_t, py::array::c_style> xadj,
        py::array_t<int32_t, py::array::c_style> adjncy,
        py::array_t<int64_t, py::array::c_style> adjwgt,
        double time_limit) {

    int n = static_cast<int>(xadj.size() - 1);
    if (n == 0) {
        return std::make_tuple(0LL, py::array_t<int32_t>(0));
    }

    OutputSuppressor suppressor;

    MaxCutGraph G_orig = build_graph(xadj, adjncy, adjwgt);
    MaxCutGraph G = G_orig;

    G.ExecuteExhaustiveKernelizationExternalsSupport({});

    // beta(G') = beta(G) + offset  =>  beta(G) = beta(G') - offset
    double raw_offset = G.GetInflictedCutChangeToKernelized();
    double cut_offset = (raw_offset <= -1e-15) ? 0.0 : raw_offset;

    auto [solver_cut, solver_partition] = G.ComputeMaxCutWithMQLib(time_limit);

    long long cut_value = static_cast<long long>(solver_cut) -
                          static_cast<long long>(cut_offset);

    // If kernelization fully reduced the graph, fall back to local search
    // on the original graph to get a partition.
    if (solver_partition.empty() || G.GetRealNumNodes() == 0) {
        auto [ls_cut, ls_part] = G_orig.ComputeLocalSearchCut();
        cut_value = std::max(cut_value, static_cast<long long>(ls_cut));
        solver_partition = ls_part;
    }

    py::array_t<int32_t> partition(n);
    auto part = partition.mutable_unchecked<1>();
    for (int i = 0; i < n; i++) part(i) = 0;
    for (int i = 0; i < static_cast<int>(solver_partition.size()) && i < n; i++) {
        part(i) = static_cast<int32_t>(solver_partition[i]);
    }

    return std::make_tuple(cut_value, partition);
}

// Kernelize + exact solve via linear kernel + brute-force on marked set.
// Returns (cut_value, partition_array).
static std::tuple<long long, py::array_t<int32_t>>
py_maxcut_exact(
        py::array_t<int64_t, py::array::c_style> xadj,
        py::array_t<int32_t, py::array::c_style> adjncy,
        py::array_t<int64_t, py::array::c_style> adjwgt,
        int time_limit) {

    int n = static_cast<int>(xadj.size() - 1);
    if (n == 0) {
        return std::make_tuple(0LL, py::array_t<int32_t>(0));
    }

    OutputSuppressor suppressor;

    MaxCutGraph G_orig = build_graph(xadj, adjncy, adjwgt);
    MaxCutGraph G = G_orig;

    // Two-way kernelization
    G.ExecuteExhaustiveKernelizationExternalsSupport({});
    double raw_offset = G.GetInflictedCutChangeToKernelized();
    double cut_offset = (raw_offset <= -1e-15) ? 0.0 : raw_offset;

    // Linear kernel: compute marked vertex set S s.t. G-S is clique forest
    G.ExecuteLinearKernelization();
    G.Algorithm2MarkedComputation();
    auto S = G.Algorithm3MarkedComputation();
    G.SetMarkedVertices(S);

    // Brute-force optimal coloring on S, extend to G-S
    auto [exact_cut, solve_time] = G.GetMaxCutWithMarkedVertexSet(
        static_cast<int>(S.size()), time_limit);

    long long cut_value = static_cast<long long>(exact_cut) -
                          static_cast<long long>(cut_offset);

    // Get a partition from local search on the original graph
    auto [ls_cut, ls_part] = G_orig.ComputeLocalSearchCut();
    cut_value = std::max(cut_value, static_cast<long long>(ls_cut));

    py::array_t<int32_t> partition(n);
    auto part = partition.mutable_unchecked<1>();
    for (int i = 0; i < n; i++) part(i) = 0;
    for (int i = 0; i < static_cast<int>(ls_part.size()) && i < n; i++) {
        part(i) = static_cast<int32_t>(ls_part[i]);
    }

    return std::make_tuple(cut_value, partition);
}

PYBIND11_MODULE(_maxcut, m) {
    m.doc() = "Python bindings for FPT Max-Cut kernelization and solvers";

    m.def("maxcut_heuristic", &py_maxcut_heuristic,
          py::arg("xadj"), py::arg("adjncy"), py::arg("adjwgt"),
          py::arg("time_limit"),
          "Kernelize + heuristic max-cut. Returns (cut_value, partition).");

    m.def("maxcut_exact", &py_maxcut_exact,
          py::arg("xadj"), py::arg("adjncy"), py::arg("adjwgt"),
          py::arg("time_limit"),
          "Kernelize + exact max-cut via FPT. Returns (cut_value, partition).");
}
