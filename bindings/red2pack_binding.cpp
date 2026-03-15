/**
 * red2pack_binding.cpp
 *
 * pybind11 binding for red2pack maximum (weighted) 2-packing set solver.
 *
 * Exposes:
 *   - solve_two_packing()         → one-shot solver with algorithm dispatch
 *   - TwoPackingKernelWrapper     → two-step reduce-and-transform + lift
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

// red2pack core
#include "red2pack/data_structure/m2s_graph_access.h"
#include "red2pack/m2s_config.h"
#include "red2pack/algorithms/rnt_solver_scheme.h"
#include "red2pack/algorithms/kernel/reduce_and_transform.h"
#include "red2pack/tools/timer.h"
#include "red2pack/tools/m2s_log.h"
#include "m2s_configuration_m2s.h"
#include "m2s_configuration_mis.h"

// Solver headers
#include "red2pack-kamis-wmis/algorithms/branch_and_reduce.h"
#include "red2pack-kamis-wmis/algorithms/weighted_rnt_exact.h"
#include "red2pack-chils/algorithms/weighted_rnt_chils.h"
#include "red2pack-chils/algorithms/drp.h"
#include "red2pack-htwis-hils/algorithms/weighted_rnt_htwis.h"
#include "red2pack-htwis-hils/algorithms/weighted_rnt_hils.h"
#include "red2pack-mmwis/algorithms/weighted_rnt_mmwis.h"
#include "red2pack-onlinemis/algorithms/heuristic.h"

// KaMIS MISConfig (from wmis)
#include "mis_config.h"

// HILS config
#include "red2pack-htwis-hils/hils_config.h"

// onlinemis config (use onlinemis/ prefix to avoid wmis configuration_mis.h)
#include "onlinemis/configuration_mis.h"

namespace py = pybind11;

// Subclass to expose protected members for the two-step API
class exposed_rnt : public red2pack::reduce_and_transform {
public:
    using reduce_and_transform::reduce_and_transform;  // inherit constructors
    using reduce_and_transform::former_node_id;
    using reduce_and_transform::reducer;
    using reduce_and_transform::solution_status;
    using reduce_and_transform::solution_offset_weight;
    using reduce_and_transform::m2s_cfg;
};

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

static std::unique_ptr<red2pack::m2s_graph_access> build_m2s_graph(
    const py::array_t<int32_t> &xadj_arr,
    const py::array_t<int32_t> &adjncy_arr,
    const py::array_t<int32_t> &vwgt_arr)
{
    auto xadj = xadj_arr.unchecked<1>();
    auto adjncy = adjncy_arr.unchecked<1>();

    int n = static_cast<int>(xadj_arr.size() - 1);
    int m = static_cast<int>(adjncy_arr.size());

    auto G = std::make_unique<red2pack::m2s_graph_access>();

    std::vector<int> xadj_vec(n + 1);
    std::vector<int> adj_vec(m);
    for (int i = 0; i <= n; i++) xadj_vec[i] = xadj(i);
    for (int i = 0; i < m; i++) adj_vec[i] = adjncy(i);

    bool has_weights = (vwgt_arr.size() == n);

    if (has_weights) {
        auto vwgt = vwgt_arr.unchecked<1>();
        std::vector<int> wgt_vec(n);
        for (int i = 0; i < n; i++) wgt_vec[i] = vwgt(i);
        std::vector<int> adjwgt(m, 1);
        G->build_from_metis_weighted(n, xadj_vec.data(), adj_vec.data(),
                                     wgt_vec.data(), adjwgt.data());
    } else {
        G->build_from_metis(n, xadj_vec.data(), adj_vec.data());
    }

    return G;
}

static void configure_m2s(red2pack::M2SConfig &cfg, double time_limit, int seed,
                           const std::string &reduction_style, bool weighted) {
    m2s_configuration_m2s configurator;
    if (weighted) {
        configurator.standard_weighted(cfg);
    } else {
        configurator.standard_unweighted(cfg);
    }

    cfg.time_limit = time_limit;
    cfg.seed = seed;
    cfg.silent = true;
    cfg.write_result = false;
    cfg.write_transformed = false;
    cfg.console_log = false;

    if (reduction_style == "fast") {
        cfg.reduction_style = red2pack::M2SConfig::Reduction_Style2::fast;
    } else if (reduction_style == "full") {
        cfg.reduction_style = red2pack::M2SConfig::Reduction_Style2::full;
    } else if (reduction_style == "strong") {
        cfg.reduction_style = red2pack::M2SConfig::Reduction_Style2::strong;
    } else if (reduction_style == "heuristic") {
        cfg.reduction_style = red2pack::M2SConfig::Reduction_Style2::heuristic;
    }
    // else: keep default from configurator
}

// --------------------------------------------------------- one-shot solver

static py::tuple solve_two_packing(
    py::array_t<int32_t> xadj,
    py::array_t<int32_t> adjncy,
    py::array_t<int32_t> vwgt,
    const std::string &algorithm,
    double time_limit,
    int seed,
    const std::string &reduction_style)
{
    auto G = build_m2s_graph(xadj, adjncy, vwgt);
    red2pack::NodeID original_n = G->number_of_nodes();

    // Weighted algorithms need standard_weighted config even on unweighted graphs,
    // because they use use_weighted_reductions() which checks disable_* flags.
    // "exact" and "online" use unweighted reductions.
    bool use_weighted = (algorithm != "exact" && algorithm != "online");

    red2pack::M2SConfig m2s_cfg{};  // value-init to zero all bools
    configure_m2s(m2s_cfg, time_limit, seed, reduction_style, use_weighted);

    red2pack::timer t;
    red2pack::NodeWeight total_weight = 0;

    std::unique_ptr<red2pack::rnt_solver_scheme> solver;

    if (algorithm == "exact") {
        MISConfig mis_cfg;
        m2s_configuration_mis().standard(mis_cfg);
        mis_cfg.seed = seed;
        mis_cfg.time_limit = time_limit;
        mis_cfg.console_log = false;
        solver = std::make_unique<red2pack::branch_and_reduce>(std::move(G), m2s_cfg, mis_cfg);
    } else if (algorithm == "exact_weighted") {
        MISConfig mis_cfg;
        m2s_configuration_mis().standard(mis_cfg);
        mis_cfg.seed = seed;
        mis_cfg.time_limit = time_limit;
        mis_cfg.console_log = false;
        solver = std::make_unique<red2pack::weighted_rnt_exact>(std::move(G), m2s_cfg, mis_cfg);
    } else if (algorithm == "chils") {
        solver = std::make_unique<red2pack::weighted_rnt_chils>(std::move(G), m2s_cfg);
    } else if (algorithm == "drp") {
        solver = std::make_unique<red2pack::drp>(std::move(G), m2s_cfg);
    } else if (algorithm == "htwis") {
        solver = std::make_unique<red2pack::weighted_rnt_htwis>(std::move(G), m2s_cfg);
    } else if (algorithm == "hils") {
        red2pack::HilsConfig hils_cfg{};
        hils_cfg.rand_seed = seed;
        hils_cfg.iterations = 1000;
        hils_cfg.p[0] = 0.55; hils_cfg.p[1] = 0.80;
        hils_cfg.p[2] = 0.55; hils_cfg.p[3] = 0.58;
        MISConfig mis_cfg;
        m2s_configuration_mis().standard(mis_cfg);
        mis_cfg.seed = seed;
        mis_cfg.time_limit = time_limit;
        mis_cfg.console_log = false;
        solver = std::make_unique<red2pack::weighted_rnt_hils>(std::move(G), m2s_cfg, hils_cfg, mis_cfg);
    } else if (algorithm == "mmwis") {
        ::mmwis::MISConfig mmwis_cfg;
        mmwis_cfg.seed = seed;
        mmwis_cfg.time_limit = time_limit;
        mmwis_cfg.console_log = false;
        solver = std::make_unique<red2pack::weighted_rnt_mmwis>(std::move(G), m2s_cfg, mmwis_cfg);
    } else if (algorithm == "online") {
        onlinemis::MISConfig omis_cfg;
        onlinemis::configuration_mis().standard(omis_cfg);
        omis_cfg.seed = seed;
        omis_cfg.time_limit = time_limit;
        omis_cfg.console_log = false;
        solver = std::make_unique<red2pack::heuristic>(std::move(G), m2s_cfg, omis_cfg);
    } else {
        throw std::invalid_argument("Unknown algorithm: " + algorithm);
    }

    // Initialize the singleton logger — allocates internal vectors like
    // reduced_nodes_mw2ps which are accessed during weighted reductions.
    red2pack::m2s_log::instance()->set_config(m2s_cfg);
    red2pack::m2s_log::instance()->set_graph(*(solver->detach()));

    {
        OutputSuppressor suppress;
        solver->solve(t);
    }

    total_weight = solver->get_solution_size();
    const auto &solution = solver->get_solution();

    std::vector<int32_t> result_verts;
    for (red2pack::NodeID i = 0; i < original_n; ++i) {
        if (solution[i]) {
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

// --------------------------------------------------------- two-step kernel wrapper

class TwoPackingKernelWrapper {
public:
    TwoPackingKernelWrapper(
        py::array_t<int32_t> xadj,
        py::array_t<int32_t> adjncy,
        py::array_t<int32_t> vwgt,
        const std::string &reduction_style,
        double time_limit,
        int seed,
        bool weighted)
    : original_n_(static_cast<red2pack::NodeID>(xadj.size() - 1)),
      reduced_(false)
    {
        G_ = build_m2s_graph(xadj, adjncy, vwgt);
        // Save original weights for computing total weight in lift_solution
        G_orig_weights_.resize(original_n_);
        for (red2pack::NodeID i = 0; i < original_n_; ++i) {
            G_orig_weights_[i] = G_->getNodeWeight(i);
        }
        configure_m2s(m2s_cfg_, time_limit, seed, reduction_style, weighted);
        // Initialize logger
        red2pack::m2s_log::instance()->set_config(m2s_cfg_);
        red2pack::m2s_log::instance()->set_graph(*G_);
        rnt_ = std::make_unique<exposed_rnt>(std::move(G_), m2s_cfg_);
    }

    // Returns (solved, kernel_xadj, kernel_adjncy, kernel_vwgt, offset_weight, kernel_n)
    py::tuple run_reduce_and_transform() {
        if (reduced_) {
            throw std::runtime_error("Already reduced. Create a new instance.");
        }

        bool solved;
        red2pack::m2s_graph_access *reduced_graph_ptr;
        {
            OutputSuppressor suppress;
            auto [s, rg] = rnt_->run_reduce_and_transform();
            solved = s;
            reduced_graph_ptr = &rg;
        }
        reduced_ = true;
        reduced_graph_ptr_ = reduced_graph_ptr;

        offset_weight_ = static_cast<long long>(rnt_->get_solution_offset_size());
        kernel_n_ = reduced_graph_ptr->number_of_nodes();

        if (solved || kernel_n_ == 0) {
            py::array_t<int64_t> out_xadj(1);
            out_xadj.mutable_at(0) = 0;
            return py::make_tuple(
                true,
                out_xadj,
                py::array_t<int32_t>(0),
                py::array_t<int32_t>(0),
                offset_weight_,
                0);
        }

        // Extract CSR from reduced graph (1-edges + 2-edges = MIS graph)
        auto &rg = *reduced_graph_ptr;
        // Count total edges
        red2pack::EdgeID total_edges = 0;
        for (red2pack::NodeID i = 0; i < kernel_n_; ++i) {
            total_edges += rg.getNodeDegree(i) + rg.getLinkDegree(i);
        }

        py::array_t<int64_t> out_xadj(kernel_n_ + 1);
        py::array_t<int32_t> out_adjncy(total_edges);
        py::array_t<int32_t> out_vwgt(kernel_n_);

        auto xp = out_xadj.mutable_data();
        auto ap = out_adjncy.mutable_data();
        auto wp = out_vwgt.mutable_data();

        red2pack::EdgeID eidx = 0;
        for (red2pack::NodeID i = 0; i < kernel_n_; ++i) {
            xp[i] = static_cast<int64_t>(eidx);
            wp[i] = static_cast<int32_t>(rg.getNodeWeight(i));
            forall_out_edges(rg, e, i) {
                ap[eidx++] = static_cast<int32_t>(rg.getEdgeTarget(e));
            } endfor
            forall_out_links(rg, e, i) {
                ap[eidx++] = static_cast<int32_t>(rg.getLinkTarget(e));
            } endfor
        }
        xp[kernel_n_] = static_cast<int64_t>(eidx);

        return py::make_tuple(
            false, out_xadj, out_adjncy, out_vwgt,
            offset_weight_, static_cast<int>(kernel_n_));
    }

    // Accept MIS vertex IDs on kernel graph, lift to original 2-packing
    py::tuple lift_solution(py::array_t<int32_t> mis_vertices) {
        if (!reduced_) {
            throw std::runtime_error("Must call run_reduce_and_transform() first.");
        }

        auto kv = mis_vertices.unchecked<1>();
        auto *ernt = static_cast<exposed_rnt*>(rnt_.get());
        auto &solution = ernt->solution_status;
        auto &fid = ernt->former_node_id;

        // Map kernel MIS vertices back to original graph via former_node_id
        red2pack::NodeWeight mis_weight = 0;
        if (kernel_n_ > 0) {
            auto &rg = *reduced_graph_ptr_;
            for (ssize_t i = 0; i < kv.shape(0); ++i) {
                int vid = kv(i);
                if (vid >= 0 && vid < static_cast<int>(kernel_n_)) {
                    red2pack::NodeID orig = fid[vid];
                    if (!solution[orig]) {
                        solution[orig] = true;
                        mis_weight += rg.getNodeWeight(vid);
                    }
                }
            }
        }

        // If reducer exists, lift reduced solution to original graph
        if (ernt->reducer) {
            ernt->reducer->build_solution(solution);
        }

        // Collect result
        std::vector<int32_t> result_verts;
        long long total_weight = 0;
        for (red2pack::NodeID i = 0; i < original_n_; ++i) {
            if (solution[i]) {
                result_verts.push_back(static_cast<int32_t>(i));
                total_weight += G_orig_weights_[i];
            }
        }

        py::array_t<int32_t> verts(result_verts.size());
        if (!result_verts.empty()) {
            std::memcpy(verts.mutable_data(), result_verts.data(),
                        result_verts.size() * sizeof(int32_t));
        }
        return py::make_tuple(total_weight, verts);
    }

    long long offset_weight() const { return offset_weight_; }
    int kernel_nodes() const { return static_cast<int>(kernel_n_); }

private:
    std::unique_ptr<red2pack::m2s_graph_access> G_;
    std::vector<red2pack::NodeWeight> G_orig_weights_;
    red2pack::M2SConfig m2s_cfg_{};  // value-init
    std::unique_ptr<exposed_rnt> rnt_;
    red2pack::NodeID original_n_;
    red2pack::NodeID kernel_n_ = 0;
    long long offset_weight_ = 0;
    bool reduced_;
    red2pack::m2s_graph_access *reduced_graph_ptr_ = nullptr;
};

// ----------------------------------------------------------- pybind11 module

PYBIND11_MODULE(_red2pack, m) {
    m.doc() = "red2pack: Maximum (weighted) 2-packing set solver";

    m.def("solve_two_packing", &solve_two_packing,
          py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
          py::arg("algorithm"), py::arg("time_limit"),
          py::arg("seed"), py::arg("reduction_style"));

    py::class_<TwoPackingKernelWrapper>(m, "TwoPackingKernel")
        .def(py::init<
            py::array_t<int32_t>,
            py::array_t<int32_t>,
            py::array_t<int32_t>,
            const std::string &,
            double, int, bool
        >(),
            py::arg("xadj"), py::arg("adjncy"), py::arg("vwgt"),
            py::arg("reduction_style"), py::arg("time_limit"),
            py::arg("seed"), py::arg("weighted"))
        .def("run_reduce_and_transform", &TwoPackingKernelWrapper::run_reduce_and_transform)
        .def("lift_solution", &TwoPackingKernelWrapper::lift_solution,
             py::arg("mis_vertices"))
        .def_property_readonly("offset_weight", &TwoPackingKernelWrapper::offset_weight)
        .def_property_readonly("kernel_nodes", &TwoPackingKernelWrapper::kernel_nodes);
}
