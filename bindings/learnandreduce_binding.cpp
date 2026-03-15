/**
 * learnandreduce_binding.cpp
 *
 * pybind11 binding for LearnAndReduce GNN-guided MWIS kernelization.
 *
 * Exposes a LearnAndReduceWrapper class with persistent C++ state:
 *   - kernelize()       → returns reduced kernel as CSR arrays
 *   - lift_solution()   → maps kernel IS back to original graph
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef __unix__
#include <fcntl.h>
#include <unistd.h>
#endif

// Global model path override — set from Python before constructing reduce_algorithm
std::string g_lr_models_path;

#include "definitions.h"
#include "graph_access.h"
#include "reduction_config.h"
#include "configuration_reduction.h"
#include "reduce_algorithm.h"

namespace py = pybind11;

// ------------------------------------------------------------------ helpers

struct OutputSuppressor {
#ifdef __unix__
    int saved_stdout = -1, saved_stderr = -1;
    OutputSuppressor() {
        std::fflush(stdout);
        std::fflush(stderr);
        saved_stdout = ::dup(STDOUT_FILENO);
        saved_stderr = ::dup(STDERR_FILENO);
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
    }
    ~OutputSuppressor() {
        std::fflush(stdout);
        std::fflush(stderr);
        if (saved_stdout >= 0) { ::dup2(saved_stdout, STDOUT_FILENO); ::close(saved_stdout); }
        if (saved_stderr >= 0) { ::dup2(saved_stderr, STDERR_FILENO); ::close(saved_stderr); }
    }
#else
    OutputSuppressor() {}
#endif
};

static void build_graph(graph_access &G,
                        const py::array_t<uint32_t> &xadj_arr,
                        const py::array_t<uint32_t> &adjncy_arr,
                        const py::array_t<uint64_t> &vwgt_arr) {
    auto xadj   = xadj_arr.unchecked<1>();
    auto adjncy = adjncy_arr.unchecked<1>();
    auto vwgt   = vwgt_arr.unchecked<1>();

    NodeID n = static_cast<NodeID>(xadj_arr.size() - 1);
    EdgeID m = static_cast<EdgeID>(adjncy_arr.size());

    // Sort adjacency lists (LearnAndReduce requires sorted neighbors)
    std::vector<NodeID> adj_copy(m);
    std::memcpy(adj_copy.data(), adjncy_arr.data(), m * sizeof(uint32_t));
    for (NodeID v = 0; v < n; ++v) {
        std::sort(adj_copy.begin() + xadj(v), adj_copy.begin() + xadj(v + 1));
    }

    std::vector<EdgeWeight> adjwgt(m, 1);
    std::vector<NodeWeight> vw(n);
    for (NodeID i = 0; i < n; ++i) vw[i] = static_cast<NodeWeight>(vwgt(i));

    // xadj is EdgeID* (uint32_t*) in LearnAndReduce
    std::vector<EdgeID> xadj_copy(n + 1);
    for (NodeID i = 0; i <= n; ++i) xadj_copy[i] = static_cast<EdgeID>(xadj(i));

    G.build_from_metis_weighted(n, xadj_copy.data(), adj_copy.data(),
                                vw.data(), adjwgt.data());
}

// --------------------------------------------------------- wrapper class

class LearnAndReduceWrapper {
public:
    LearnAndReduceWrapper(
        py::array_t<uint32_t> xadj,
        py::array_t<uint32_t> adjncy,
        py::array_t<uint64_t> vwgt,
        const std::string &config_name,
        const std::string &gnn_filter,
        double time_limit,
        int seed,
        const std::string &models_path)
    : G_(new graph_access()),
      original_n_(static_cast<NodeID>(xadj.size() - 1)),
      kernelized_(false)
    {
        build_graph(*G_, xadj, adjncy, vwgt);

        // Configure
        configuration_reduction configurator;
        configurator.all_reductions(config_);

        if (config_name == "cyclic_strong") {
            configurator.cyclicStrong(config_);
            config_.struction_config_name = "cyclicStrong";
        } else {
            // default: cyclic_fast
            configurator.cyclicFast(config_);
            config_.struction_config_name = "cyclicFast";
        }

        config_.time_limit = time_limit;
        config_.seed = seed;
        config_.console_log = false;
        config_.verbose = false;
        config_.print_log = false;
        config_.write_kernel = false;
        config_.write_solution = false;
        config_.print_reduction_info = false;
        config_.weight_source = ReductionConfig::Weight_Source::FILE;

        config_.setGNNFilterStyle(gnn_filter);

        // Set global model path for get_models_folder_path() override
        g_lr_models_path = models_path;

        reducer_ = std::make_unique<reduce_algorithm>(*G_, config_);
    }

    // Returns (kernel_xadj, kernel_adjncy, kernel_vwgt, offset_weight, kernel_n)
    py::tuple kernelize() {
        if (kernelized_) {
            throw std::runtime_error("Already kernelized. Create a new instance.");
        }

        OutputSuppressor suppress;
        graph_access &kernel = reducer_->kernelize();
        kernelized_ = true;
        kernel_n_ = kernel.number_of_nodes();

        long long offset = static_cast<long long>(reducer_->get_current_is_weight());

        if (kernel_n_ == 0) {
            py::array_t<int64_t> xadj(1);
            xadj.mutable_at(0) = 0;
            return py::make_tuple(
                xadj,
                py::array_t<int32_t>(0),
                py::array_t<int64_t>(0),
                offset,
                0);
        }

        EdgeID kernel_m = kernel.number_of_edges();

        py::array_t<int64_t> out_xadj(kernel_n_ + 1);
        py::array_t<int32_t> out_adjncy(kernel_m);
        py::array_t<int64_t> out_vwgt(kernel_n_);

        auto xp = out_xadj.mutable_data();
        auto ap = out_adjncy.mutable_data();
        auto wp = out_vwgt.mutable_data();

        EdgeID eidx = 0;
        for (NodeID i = 0; i < kernel_n_; ++i) {
            xp[i] = static_cast<int64_t>(eidx);
            wp[i] = static_cast<int64_t>(kernel.getNodeWeight(i));
            for (EdgeID e = kernel.get_first_edge(i); e < kernel.get_first_invalid_edge(i); ++e) {
                ap[eidx++] = static_cast<int32_t>(kernel.getEdgeTarget(e));
            }
        }
        xp[kernel_n_] = static_cast<int64_t>(eidx);

        return py::make_tuple(out_xadj, out_adjncy, out_vwgt, offset, static_cast<int>(kernel_n_));
    }

    // Accept kernel IS vertex IDs, lift to original, return (total_weight, vertices)
    py::tuple lift_solution(py::array_t<int32_t> kernel_vertices) {
        if (!kernelized_) {
            throw std::runtime_error("Must call kernelize() first.");
        }

        auto kv = kernel_vertices.unchecked<1>();

        std::vector<bool> reduced_sol(kernel_n_, false);
        for (ssize_t i = 0; i < kv.shape(0); ++i) {
            int vid = kv(i);
            if (vid >= 0 && vid < static_cast<int>(kernel_n_)) {
                reduced_sol[vid] = true;
            }
        }

        std::vector<bool> full_sol(original_n_, false);

        NodeWeight total_weight;
        {
            OutputSuppressor suppress;
            total_weight = reducer_->lift_solution(reduced_sol, full_sol);
        }

        std::vector<int32_t> result_verts;
        result_verts.reserve(original_n_ / 4);
        for (NodeID i = 0; i < original_n_; ++i) {
            if (full_sol[i]) {
                result_verts.push_back(static_cast<int32_t>(i));
            }
        }

        py::array_t<int32_t> verts(result_verts.size());
        if (!result_verts.empty()) {
            std::memcpy(verts.mutable_data(), result_verts.data(),
                        result_verts.size() * sizeof(int32_t));
        }

        return py::make_tuple(static_cast<long long>(total_weight), verts);
    }

    long long offset_weight() const {
        return static_cast<long long>(reducer_->get_current_is_weight());
    }

private:
    // G_ must outlive reducer_ (destroy reducer first, then G)
    std::unique_ptr<graph_access> G_;
    ReductionConfig config_;
    std::unique_ptr<reduce_algorithm> reducer_;
    NodeID original_n_;
    NodeID kernel_n_ = 0;
    bool kernelized_;
};

// ----------------------------------------------------------- pybind11 module

PYBIND11_MODULE(_learnandreduce, m) {
    m.doc() = "LearnAndReduce GNN-guided MWIS kernelization";

    py::class_<LearnAndReduceWrapper>(m, "LearnAndReduceKernel")
        .def(py::init<
            py::array_t<uint32_t>,
            py::array_t<uint32_t>,
            py::array_t<uint64_t>,
            const std::string &,
            const std::string &,
            double, int,
            const std::string &
        >(),
            py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
            py::arg("config"), py::arg("gnn_filter"),
            py::arg("time_limit"), py::arg("seed"),
            py::arg("models_path"))
        .def("kernelize", &LearnAndReduceWrapper::kernelize)
        .def("lift_solution", &LearnAndReduceWrapper::lift_solution,
             py::arg("kernel_vertices"))
        .def_property_readonly("offset_weight", &LearnAndReduceWrapper::offset_weight);
}
