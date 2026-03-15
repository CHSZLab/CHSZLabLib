// pybind11 binding for HeiCut — exact minimum cut on hypergraphs.
// Exposes four algorithms: kernelizer, ilp, trimmer, submodular.
// Hypergraphs are passed as CSR arrays (matching CHSZLabLib's HyperGraph).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>

// Mt-KaHyPar headers
#include "mt-kahypar-library/libmtkahypar.h"
#include "mt-kahypar/utils/cast.h"

// HeiCut headers
#include "lib/utils/definitions.h"
#include "lib/utils/const.h"
#include "lib/utils/random.h"
#include "lib/parse_parameters/parse_parameters.h"
#include "lib/solvers/kernelizer.h"
#include "lib/solvers/ilp.h"
#include "lib/solvers/submodular.h"
#include "lib/orderer/orderer.h"
#include "lib/trimmer/trimmer.h"
#include "lib/io/mt_kahypar_io.h"

// Gurobi shim (resolved via include path to shims/gurobi/)
#include "gurobi_c++.h"

// KaHIP timer
#include "kahip/timer.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: write CSR hypergraph to a temp file in hMetis format
// hMetis format:
//   Line 1:  num_edges  num_nodes  [fmt]
//   fmt=1 → edge weights only; fmt=10 → node weights only; fmt=11 → both
//   Next num_edges lines: [edge_weight] node1 node2 ... (1-indexed)
//   Next num_nodes lines (if node weights): node_weight
// ---------------------------------------------------------------------------
static std::string write_hmetis_tmpfile(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int64_t, py::array::c_style> node_weights,
    py::array_t<int64_t, py::array::c_style> edge_weights)
{
    auto ep = eptr.unchecked<1>();
    auto ev = everts.unchecked<1>();

    int64_t num_edges = ep.shape(0) - 1;
    // Find num_nodes as max vertex id + 1
    int32_t max_node = 0;
    for (ssize_t i = 0; i < ev.shape(0); ++i)
        if (ev(i) > max_node) max_node = ev(i);
    int64_t num_nodes = max_node + 1;

    bool has_ew = edge_weights.size() > 0;
    bool has_nw = node_weights.size() > 0;

    int fmt = 0;
    if (has_ew && has_nw)  fmt = 11;
    else if (has_ew)       fmt = 1;
    else if (has_nw)       fmt = 10;

    // Create temp file
    char tmpname[] = "/tmp/heicut_XXXXXX.hgr";
    int fd = mkstemps(tmpname, 4);
    if (fd < 0) throw std::runtime_error("Failed to create temp file");

    FILE* f = fdopen(fd, "w");
    if (!f) { close(fd); throw std::runtime_error("Failed to open temp file"); }

    // Header
    if (fmt > 0)
        fprintf(f, "%ld %ld %d\n", (long)num_edges, (long)num_nodes, fmt);
    else
        fprintf(f, "%ld %ld\n", (long)num_edges, (long)num_nodes);

    // Edge lines
    const int64_t* ew_data = has_ew ? edge_weights.data() : nullptr;
    for (int64_t e = 0; e < num_edges; ++e) {
        if (has_ew) fprintf(f, "%ld ", (long)ew_data[e]);
        for (int64_t p = ep(e); p < ep(e + 1); ++p) {
            if (p > ep(e)) fputc(' ', f);
            fprintf(f, "%d", ev(p) + 1); // 1-indexed
        }
        fputc('\n', f);
    }

    // Node weight lines
    if (has_nw) {
        auto nw = node_weights.unchecked<1>();
        for (int64_t n = 0; n < num_nodes; ++n)
            fprintf(f, "%ld\n", (long)nw(n));
    }

    fclose(f);
    return std::string(tmpname);
}

// ---------------------------------------------------------------------------
// Suppress stdout/stderr at fd level
// ---------------------------------------------------------------------------
struct FDSuppressor {
    int old_stdout, old_stderr;
    FDSuppressor() {
        fflush(stdout); fflush(stderr);
        old_stdout = dup(STDOUT_FILENO);
        old_stderr = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    ~FDSuppressor() {
        fflush(stdout); fflush(stderr);
        dup2(old_stdout, STDOUT_FILENO);
        dup2(old_stderr, STDERR_FILENO);
        close(old_stdout);
        close(old_stderr);
    }
};

// ---------------------------------------------------------------------------
// kernelizer
// ---------------------------------------------------------------------------
static std::tuple<uint32_t, double>
py_heicut_kernelizer(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int64_t, py::array::c_style> node_weights,
    py::array_t<int64_t, py::array::c_style> edge_weights,
    const std::string& base_solver_str,
    double ilp_timeout,
    int seed,
    int threads,
    bool unweighted)
{
    std::string tmpfile = write_hmetis_tmpfile(eptr, everts, node_weights, edge_weights);

    KernelizerConfig config{};
    config.hypergraphFileName = tmpfile.c_str();
    config.hypergraphFileFormat = HMETIS;
    config.seed = seed;
    config.presetType = DETERMINISTIC;
    config.numThreads = threads;
    config.unweighted = unweighted;
    config.baseSolver = (base_solver_str == "ilp") ? BaseSolver::ILP : BaseSolver::SUBMODULAR;
    config.ilpTimeout = ilp_timeout;
    config.ilpMode = DEFAULT_ILP_MODE;
    config.orderingType = DEFAULT_ORDERING_TYPE;
    config.orderingMode = DEFAULT_ORDERING_MODE;
    config.pruningMode = DEFAULT_PRUNING_MODE;
    config.LPNumIterations = DEFAULT_LP_NUM_ITERATIONS;
    config.LPMode = DEFAULT_LP_MODE;
    config.LPNumPinsToSample = DEFAULT_LP_NUM_PINS_TO_SAMPLE;
    config.verbose = false;

    CutValue minEdgeCut = 0;
    double totalTime = 0.0;

    {
        FDSuppressor suppress;
        RandomFunctions::set_seed(config.seed);

        mt_kahypar_hypergraph_t wrapper = MtKaHyParIO::read_hypergraph_from_file(config);
        StaticHypergraph& hg = mt_kahypar::utils::cast<StaticHypergraph>(wrapper);

        EdgeIndex inputNumEdges = hg.initialNumEdges() - hg.numRemovedHyperedges();
        NodeIndex inputNumNodes = hg.initialNumNodes() - hg.numRemovedHypernodes();

        if (config.unweighted) {
            for (EdgeID e : hg.edges())
                if (hg.edgeIsEnabled(e))
                    hg.setEdgeWeight(e, 1);
        }

        Kernelizer kernelizer(config);
        KernelizerResult result = kernelizer.compute_mincut(hg, inputNumNodes, inputNumEdges);
        minEdgeCut = result.minEdgeCut;
        totalTime = result.time;

        mt_kahypar_free_hypergraph(wrapper);
    }

    std::remove(tmpfile.c_str());
    return std::make_tuple(minEdgeCut, totalTime);
}

// ---------------------------------------------------------------------------
// ilp
// ---------------------------------------------------------------------------
static std::tuple<uint32_t, double>
py_heicut_ilp(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int64_t, py::array::c_style> node_weights,
    py::array_t<int64_t, py::array::c_style> edge_weights,
    double ilp_timeout,
    const std::string& ilp_mode_str,
    int seed,
    int threads,
    bool unweighted)
{
    std::string tmpfile = write_hmetis_tmpfile(eptr, everts, node_weights, edge_weights);

    ILPConfig config{};
    config.hypergraphFileName = tmpfile.c_str();
    config.hypergraphFileFormat = HMETIS;
    config.seed = seed;
    config.presetType = DETERMINISTIC;
    config.numThreads = threads;
    config.unweighted = unweighted;
    config.ilpTimeout = ilp_timeout;
    config.ilpMode = (ilp_mode_str == "milp") ? ILPMode::MILP : ILPMode::BIP;

    CutValue minEdgeCut = 0;
    double totalTime = 0.0;

    {
        FDSuppressor suppress;

        mt_kahypar_hypergraph_t wrapper = MtKaHyParIO::read_hypergraph_from_file(config);
        StaticHypergraph& hg = mt_kahypar::utils::cast<StaticHypergraph>(wrapper);

        EdgeIndex inputNumEdges = hg.initialNumEdges() - hg.numRemovedHyperedges();
        NodeIndex inputNumNodes = hg.initialNumNodes() - hg.numRemovedHypernodes();

        if (config.unweighted) {
            for (EdgeID e : hg.edges())
                if (hg.edgeIsEnabled(e))
                    hg.setEdgeWeight(e, 1);
        }

        timer t;

        // ILP needs GIL released before optimize() reacquires it
        try {
            GRBEnv env = GRBEnv(true);
            env.start();

            ILPMincut solver(inputNumNodes, inputNumEdges, env,
                             config.seed, config.ilpTimeout,
                             config.ilpMode, config.numThreads);
            solver.add_node_variables_and_constraints(hg);
            solver.add_edge_variables_and_constraints(hg);
            solver.set_objective();
            solver.optimize();
            minEdgeCut = solver.get_result(hg);
        } catch (GRBException& e) {
            throw std::runtime_error(
                std::string("GRBException: ") + e.getMessage() +
                " (code=" + std::to_string(e.getErrorCode()) + ")");
        }

        totalTime = t.elapsed();
        mt_kahypar_free_hypergraph(wrapper);
    }

    std::remove(tmpfile.c_str());
    return std::make_tuple(minEdgeCut, totalTime);
}

// ---------------------------------------------------------------------------
// submodular
// ---------------------------------------------------------------------------
static std::tuple<uint32_t, double>
py_heicut_submodular(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int64_t, py::array::c_style> node_weights,
    py::array_t<int64_t, py::array::c_style> edge_weights,
    const std::string& ordering_type_str,
    const std::string& ordering_mode_str,
    int seed,
    int threads,
    bool unweighted)
{
    std::string tmpfile = write_hmetis_tmpfile(eptr, everts, node_weights, edge_weights);

    SubmodularConfig config{};
    config.hypergraphFileName = tmpfile.c_str();
    config.hypergraphFileFormat = HMETIS;
    config.seed = seed;
    // Submodular uses DynamicHypergraph which requires HIGHEST_QUALITY preset
    config.presetType = HIGHEST_QUALITY;
    config.numThreads = threads;
    config.unweighted = unweighted;

    if (ordering_type_str == "ma") config.orderingType = OrderingType::MA;
    else if (ordering_type_str == "queyranne") config.orderingType = OrderingType::QUEYRANNE;
    else config.orderingType = OrderingType::TIGHT;

    if (ordering_mode_str == "multi") config.orderingMode = OrderingMode::MULTI;
    else config.orderingMode = OrderingMode::SINGLE;

    CutValue minEdgeCut = 0;
    double totalTime = 0.0;

    {
        FDSuppressor suppress;
        RandomFunctions::set_seed(config.seed);

        mt_kahypar_hypergraph_t wrapper = MtKaHyParIO::read_hypergraph_from_file(config);
        DynamicHypergraph& hg = mt_kahypar::utils::cast<DynamicHypergraph>(wrapper);

        EdgeIndex inputNumEdges = hg.initialNumEdges() - hg.numRemovedHyperedges();
        NodeIndex inputNumNodes = hg.initialNumNodes() - hg.numRemovedHypernodes();

        bool hasWeightedEdges = false;
        if (config.unweighted) {
            for (EdgeID e : hg.edges())
                if (hg.edgeIsEnabled(e))
                    hg.setEdgeWeight(e, 1);
        } else {
            for (EdgeID e : hg.edges())
                if (hg.edgeIsEnabled(e) && hg.edgeWeight(e) != 1) {
                    hasWeightedEdges = true;
                    break;
                }
        }

        timer t;

        SubmodularMincut solver(hg.initialNumNodes(), inputNumEdges,
                                config.orderingType, config.orderingMode,
                                hasWeightedEdges, config.numThreads);
        SubmodularMincutResult result = solver.solve(hg);
        minEdgeCut = result.minEdgeCut;
        totalTime = t.elapsed();

        mt_kahypar_free_hypergraph(wrapper);
    }

    std::remove(tmpfile.c_str());
    return std::make_tuple(minEdgeCut, totalTime);
}

// ---------------------------------------------------------------------------
// trimmer
// ---------------------------------------------------------------------------
static std::tuple<uint32_t, double>
py_heicut_trimmer(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int64_t, py::array::c_style> node_weights,
    py::array_t<int64_t, py::array::c_style> edge_weights,
    const std::string& ordering_type_str,
    const std::string& ordering_mode_str,
    int seed,
    int threads)
{
    // Trimmer only works on unweighted hypergraphs
    std::string tmpfile = write_hmetis_tmpfile(eptr, everts, node_weights, edge_weights);

    TrimmerConfig config{};
    config.hypergraphFileName = tmpfile.c_str();
    config.hypergraphFileFormat = HMETIS;
    config.seed = seed;
    config.presetType = DETERMINISTIC;
    config.numThreads = threads;
    config.unweighted = true; // trimmer always unweighted

    if (ordering_type_str == "ma") config.orderingType = OrderingType::MA;
    else if (ordering_type_str == "queyranne") config.orderingType = OrderingType::QUEYRANNE;
    else config.orderingType = OrderingType::TIGHT;

    if (ordering_mode_str == "multi") config.orderingMode = OrderingMode::MULTI;
    else config.orderingMode = OrderingMode::SINGLE;

    CutValue minEdgeCut = 0;
    double totalTime = 0.0;

    {
        FDSuppressor suppress;
        RandomFunctions::set_seed(config.seed);

        mt_kahypar_hypergraph_t wrapper = MtKaHyParIO::read_hypergraph_from_file(config);
        StaticHypergraph& hg = mt_kahypar::utils::cast<StaticHypergraph>(wrapper);

        EdgeIndex inputNumEdges = hg.initialNumEdges() - hg.numRemovedHyperedges();
        NodeIndex inputNumNodes = hg.initialNumNodes() - hg.numRemovedHypernodes();

        // Trimmer: always set edge weights to 1
        for (EdgeID e : hg.edges())
            if (hg.edgeIsEnabled(e))
                hg.setEdgeWeight(e, 1);

        bool hasWeightedEdges = false;
        timer t;

        // MA-ordering on input hypergraph
        std::vector<NodeID> nodeOrdering(inputNumNodes);
        std::vector<EdgeID> edgeHeadOrdering(inputNumEdges);
        std::vector<NodeID> edgeHead(inputNumEdges);

        Orderer<StaticHypergraph, EdgeWeight> inputOrderer(
            hg.initialNumNodes(), inputNumEdges, OrderingType::MA,
            hasWeightedEdges, RandomFunctions::get_random_engine());
        inputOrderer.compute_ordering(hg, inputNumNodes,
                                      &nodeOrdering, &edgeHeadOrdering, &edgeHead);

        // Build trimmer
        Trimmer trimmer(hg.initialNumNodes(), inputNumNodes, inputNumEdges,
                        hg, nodeOrdering, edgeHeadOrdering, edgeHead, HIGHEST_QUALITY);
        trimmer.build_backward_edges();

        // Submodular solver for each k-trimmed certificate
        SubmodularMincut subSolver(hg.initialNumNodes(), inputNumEdges,
                                   config.orderingType, config.orderingMode,
                                   hasWeightedEdges, config.numThreads);

        // Exponential search on k
        TrimmerValue k = 2;
        while (true) {
            mt_kahypar_hypergraph_t trimmedWrapper =
                trimmer.create_k_trimmed_certificate(k);
            DynamicHypergraph& trimmedHg =
                mt_kahypar::utils::cast<DynamicHypergraph>(trimmedWrapper);

            SubmodularMincutResult result = subSolver.solve(trimmedHg);
            minEdgeCut = result.minEdgeCut;

            mt_kahypar_free_hypergraph(trimmedWrapper);

            if (k > minEdgeCut) break;
            k *= 2;
        }

        totalTime = t.elapsed();
        mt_kahypar_free_hypergraph(wrapper);
    }

    std::remove(tmpfile.c_str());
    return std::make_tuple(minEdgeCut, totalTime);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_heicut, m) {
    m.doc() = "Python bindings for HeiCut — exact minimum cut on hypergraphs";

    m.def("kernelizer", &py_heicut_kernelizer,
          py::arg("eptr"), py::arg("everts"),
          py::arg("node_weights"), py::arg("edge_weights"),
          py::arg("base_solver") = "submodular",
          py::arg("ilp_timeout") = 7200.0,
          py::arg("seed") = 0,
          py::arg("threads") = 1,
          py::arg("unweighted") = false,
          R"doc(
          Compute hypergraph minimum cut using kernelization.

          Applies coarsening/pruning rules, then solves the kernel with
          either a submodular solver or ILP.

          Parameters
          ----------
          eptr : ndarray[int64]
              Hyperedge pointers (CSR format).
          everts : ndarray[int32]
              Vertex indices for each hyperedge (CSR format).
          node_weights : ndarray[int64]
              Node weights (empty for unit weights).
          edge_weights : ndarray[int64]
              Hyperedge weights (empty for unit weights).
          base_solver : str
              Base solver: "submodular" or "ilp".
          ilp_timeout : float
              ILP time limit in seconds (only if base_solver="ilp").
          seed : int
              Random seed.
          threads : int
              Number of threads.
          unweighted : bool
              Force unweighted processing.

          Returns
          -------
          tuple[int, float]
              (min_edge_cut, computing_time).
          )doc");

    m.def("ilp", &py_heicut_ilp,
          py::arg("eptr"), py::arg("everts"),
          py::arg("node_weights"), py::arg("edge_weights"),
          py::arg("ilp_timeout") = 7200.0,
          py::arg("ilp_mode") = "bip",
          py::arg("seed") = 0,
          py::arg("threads") = 1,
          py::arg("unweighted") = false,
          R"doc(
          Compute hypergraph minimum cut via integer linear programming.

          Parameters
          ----------
          ilp_mode : str
              "bip" (binary IP) or "milp" (mixed ILP).
          )doc");

    m.def("submodular", &py_heicut_submodular,
          py::arg("eptr"), py::arg("everts"),
          py::arg("node_weights"), py::arg("edge_weights"),
          py::arg("ordering_type") = "tight",
          py::arg("ordering_mode") = "single",
          py::arg("seed") = 0,
          py::arg("threads") = 1,
          py::arg("unweighted") = false,
          R"doc(
          Compute hypergraph minimum cut via submodular optimization.

          Parameters
          ----------
          ordering_type : str
              "ma", "tight", or "queyranne".
          ordering_mode : str
              "single" or "multi".
          )doc");

    m.def("trimmer", &py_heicut_trimmer,
          py::arg("eptr"), py::arg("everts"),
          py::arg("node_weights"), py::arg("edge_weights"),
          py::arg("ordering_type") = "tight",
          py::arg("ordering_mode") = "single",
          py::arg("seed") = 0,
          py::arg("threads") = 1,
          R"doc(
          Compute exact hypergraph minimum cut using k-trimming (Chekuri-Xu).

          Only works on unweighted hypergraphs. Uses exponential search
          on k with submodular solver for each k-trimmed certificate.
          )doc");
}
