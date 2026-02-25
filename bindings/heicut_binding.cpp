/******************************************************************************
 * heicut_binding.cpp
 *
 * Pybind11 bindings for HeiCut hypergraph minimum cut solvers.
 * Exposes: kernelizer, submodular, trimmer, and (optionally) ilp.
 *****************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <cassert>
#include <algorithm>

// HeiCut headers
#include "lib/utils/definitions.h"
#include "lib/utils/const.h"
#include "lib/utils/random.h"
#include "lib/coarsening/pruner.h"
#include "lib/coarsening/label_propagation.h"
#include "lib/solvers/submodular.h"
#include "lib/orderer/orderer.h"
#include "lib/trimmer/trimmer.h"

// Mt-KaHyPar C API + cast
#include "mt-kahypar-library/libmtkahypar.h"
#include "mt-kahypar/utils/cast.h"

// KaHIP timer
#include "kahip/timer.h"

#ifdef HEICUT_HAS_GUROBI
#include "lib/solvers/kernelizer.h"
#include "lib/solvers/ilp.h"
#include "gurobi_c++.h"
#endif

namespace py = pybind11;

// ============================================================================
// Helpers
// ============================================================================

// Ensure mt-kahypar thread pool is initialized exactly once.
static void ensure_thread_pool(size_t num_threads) {
    static bool initialized = false;
    if (!initialized) {
        mt_kahypar_initialize_thread_pool(num_threads, true);
        initialized = true;
    }
}

// RAII guard to suppress stdout/stderr from HeiCut
struct OutputSuppressor {
    std::streambuf *old_cout, *old_cerr;
    std::ostringstream null_stream;
    OutputSuppressor() {
        old_cout = std::cout.rdbuf(null_stream.rdbuf());
        old_cerr = std::cerr.rdbuf(null_stream.rdbuf());
    }
    ~OutputSuppressor() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
};

// Parse ordering_type string to enum
static OrderingType parse_ordering_type(const std::string& s) {
    if (s == "ma" || s == "MA") return OrderingType::MA;
    if (s == "tight" || s == "TIGHT") return OrderingType::TIGHT;
    if (s == "queyranne" || s == "QUEYRANNE") return OrderingType::QUEYRANNE;
    throw std::invalid_argument("Unknown ordering_type: '" + s + "'. Use 'ma', 'tight', or 'queyranne'.");
}

// Build a mt-kahypar StaticHypergraph from Python CSR arrays
static mt_kahypar_hypergraph_t build_static_hypergraph(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes)
{
    auto eptr_buf = eptr.unchecked<1>();
    auto everts_buf = everts.unchecked<1>();
    int num_edges = static_cast<int>(eptr.size() - 1);
    size_t num_pins = static_cast<size_t>(eptr_buf(num_edges));

    std::unique_ptr<size_t[]> adj_indices(new size_t[num_edges + 1]);
    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> adj_list(new mt_kahypar_hyperedge_id_t[num_pins]);
    std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> ew(new mt_kahypar_hyperedge_weight_t[num_edges]);

    for (int e = 0; e < num_edges + 1; e++)
        adj_indices[e] = static_cast<size_t>(eptr_buf(e));

    for (size_t i = 0; i < num_pins; i++)
        adj_list[i] = static_cast<mt_kahypar_hyperedge_id_t>(everts_buf(i));

    bool has_weights = (edge_weights.size() > 0);
    if (has_weights) {
        auto ew_buf = edge_weights.unchecked<1>();
        for (int e = 0; e < num_edges; e++)
            ew[e] = static_cast<mt_kahypar_hyperedge_weight_t>(ew_buf(e));
    } else {
        for (int e = 0; e < num_edges; e++)
            ew[e] = 1;
    }

    return mt_kahypar_create_hypergraph(
        DEFAULT,
        static_cast<mt_kahypar_hypernode_id_t>(num_nodes),
        static_cast<mt_kahypar_hyperedge_id_t>(num_edges),
        adj_indices.get(),
        adj_list.get(),
        ew.get(),
        nullptr);
}

// Build a mt-kahypar DynamicHypergraph from Python CSR arrays
static mt_kahypar_hypergraph_t build_dynamic_hypergraph(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes)
{
    auto eptr_buf = eptr.unchecked<1>();
    auto everts_buf = everts.unchecked<1>();
    int num_edges = static_cast<int>(eptr.size() - 1);
    size_t num_pins = static_cast<size_t>(eptr_buf(num_edges));

    std::unique_ptr<size_t[]> adj_indices(new size_t[num_edges + 1]);
    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> adj_list(new mt_kahypar_hyperedge_id_t[num_pins]);
    std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> ew(new mt_kahypar_hyperedge_weight_t[num_edges]);

    for (int e = 0; e < num_edges + 1; e++)
        adj_indices[e] = static_cast<size_t>(eptr_buf(e));

    for (size_t i = 0; i < num_pins; i++)
        adj_list[i] = static_cast<mt_kahypar_hyperedge_id_t>(everts_buf(i));

    bool has_weights = (edge_weights.size() > 0);
    if (has_weights) {
        auto ew_buf = edge_weights.unchecked<1>();
        for (int e = 0; e < num_edges; e++)
            ew[e] = static_cast<mt_kahypar_hyperedge_weight_t>(ew_buf(e));
    } else {
        for (int e = 0; e < num_edges; e++)
            ew[e] = 1;
    }

    return mt_kahypar_create_hypergraph(
        HIGHEST_QUALITY,
        static_cast<mt_kahypar_hypernode_id_t>(num_nodes),
        static_cast<mt_kahypar_hyperedge_id_t>(num_edges),
        adj_indices.get(),
        adj_list.get(),
        ew.get(),
        nullptr);
}

// ============================================================================
// kernelizer: pruning + label propagation + submodular solver
// ============================================================================
// We implement the kernelizer logic directly here so it works without Gurobi.
// The kernelizer.cpp from HeiCut requires gurobi_c++.h unconditionally.
// This implementation mirrors the SUBMODULAR path of Kernelizer::compute_mincut.

static std::tuple<uint32_t, double> py_kernelizer(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes,
    const std::string& ordering_type_str,
    int lp_iterations,
    int seed,
    int num_threads)
{
    ensure_thread_pool(static_cast<size_t>(num_threads));
    OutputSuppressor suppress;

    RandomFunctions::set_seed(seed);

    // Build a static hypergraph for kernelization
    mt_kahypar_hypergraph_t hg_wrapper = build_static_hypergraph(eptr, everts, edge_weights, num_nodes);
    StaticHypergraph& hypergraph = mt_kahypar::utils::cast<StaticHypergraph>(hg_wrapper);

    OrderingType orderingType = parse_ordering_type(ordering_type_str);

    // Build KernelizerConfig equivalent
    // We use the submodular solver path (no Gurobi needed)
    KernelizerConfig config;
    config.seed = seed;
    config.numThreads = static_cast<size_t>(num_threads);
    config.baseSolver = BaseSolver::SUBMODULAR;
    config.orderingType = orderingType;
    config.orderingMode = OrderingMode::SINGLE;
    config.pruningMode = PruningMode::BEST;
    config.LPNumIterations = static_cast<IterationIndex>(lp_iterations);
    config.LPMode = LabelPropagationMode::CLIQUE_EXPANDED;
    config.LPNumPinsToSample = DEFAULT_LP_NUM_PINS_TO_SAMPLE;
    config.verbose = false;
    config.ilpTimeout = DEFAULT_TIMEOUT_ILP;
    config.ilpMode = DEFAULT_ILP_MODE;
    config.unweighted = false;
    config.presetType = DEFAULT;
    config.hypergraphFileName = nullptr;
    config.hypergraphFileFormat = HMETIS;

    NodeIndex inputNumNodes = hypergraph.initialNumNodes() - hypergraph.numRemovedHypernodes();
    EdgeIndex inputNumEdges = hypergraph.initialNumEdges() - hypergraph.numRemovedHyperedges();

#ifdef HEICUT_HAS_GUROBI
    // Use the real Kernelizer from HeiCut
    Kernelizer kernelizer(config);
    KernelizerResult result = kernelizer.compute_mincut(hypergraph, inputNumNodes, inputNumEdges);
    uint32_t cut_value = result.minEdgeCut;
    double total_time = result.time;
#else
    // Implement kernelizer logic inline using submodular solver
    timer t;
    double totalComputingTime = 0;

    Pruner pruner;

    // Naive estimate
    t.restart();
    CutValue minEdgeCut = pruner.compute_naive_mincut_estimate(hypergraph);
    totalComputingTime += t.elapsed();

    if (minEdgeCut == 0) {
        mt_kahypar_free_hypergraph(hg_wrapper);
        return std::make_tuple(static_cast<uint32_t>(0), totalComputingTime);
    }

    // Remove trivial edges
    t.restart();
    pruner.remove_hyperedges_of_size_one_or_weight_zero(hypergraph);
    totalComputingTime += t.elapsed();

    // Label propagation
    LabelPropagation labelPropagater(config.LPNumIterations, config.LPMode, config.LPNumPinsToSample);

    IterationIndex roundCounter = 0;
    NodeIndex numNodesBeforePruningRules = 0;

    auto get_num_nodes = [&]() -> NodeIndex {
        return hypergraph.initialNumNodes() - hypergraph.numRemovedHypernodes();
    };
    auto get_num_edges = [&]() -> EdgeIndex {
        return hypergraph.initialNumEdges() - hypergraph.numRemovedHyperedges();
    };
    auto update_min_cut = [&]() {
        if (get_num_nodes() == 1) return;
        CutValue naive = pruner.compute_naive_mincut_estimate(hypergraph);
        minEdgeCut = std::min(minEdgeCut, naive);
    };
    auto can_stop_early = [&]() -> bool {
        return minEdgeCut == 0 || get_num_nodes() == 1 || get_num_edges() == 0;
    };

    // Main kernelization loop
    while (roundCounter == 0 || get_num_nodes() < numNodesBeforePruningRules) {
        roundCounter++;

        // Label propagation (if iterations > 0)
        if (config.LPNumIterations > 0) {
            t.restart();
            hypergraph = labelPropagater.propagate_and_contract_labels(hypergraph);
            totalComputingTime += t.elapsed();
            update_min_cut();
            if (can_stop_early()) break;
        }

        // Heavy edges
        numNodesBeforePruningRules = get_num_nodes();
        t.restart();
        hypergraph = pruner.contract_hyperedges_not_lighter_than_estimate(hypergraph, minEdgeCut);
        totalComputingTime += t.elapsed();
        update_min_cut();
        if (can_stop_early()) break;

        // Heavy overlaps
        t.restart();
        hypergraph = pruner.contract_overlaps_not_lighter_than_estimate(hypergraph, minEdgeCut);
        totalComputingTime += t.elapsed();
        update_min_cut();
        if (can_stop_early()) break;

        // Shiftable 2-edges
        t.restart();
        hypergraph = pruner.contract_shiftable_hyperedges_of_size_two(hypergraph);
        totalComputingTime += t.elapsed();
        update_min_cut();
        if (can_stop_early()) break;

        // Triangle 2-edges
        t.restart();
        hypergraph = pruner.contract_triangle_hyperedges_of_size_two(hypergraph, minEdgeCut);
        totalComputingTime += t.elapsed();
        update_min_cut();
        if (can_stop_early()) break;
    }

    if (!can_stop_early()) {
        // Need submodular solver on the remaining kernel
        // Convert static to dynamic hypergraph
        t.restart();
        const NodeIndex numNodes2 = hypergraph.initialNumNodes();
        const EdgeIndex numEdges2 = hypergraph.initialNumEdges();
        const NodeIndex numPins2 = hypergraph.initialNumPins();
        bool hasWeightedEdges = false;

        std::unique_ptr<size_t[]> eai(new size_t[numEdges2 + 1]);
        std::unique_ptr<mt_kahypar_hyperedge_id_t[]> eal(new mt_kahypar_hyperedge_id_t[numPins2]);
        std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> ewt(new mt_kahypar_hyperedge_weight_t[numEdges2]);
        std::unique_ptr<mt_kahypar_hypernode_weight_t[]> nwt(new mt_kahypar_hypernode_weight_t[numNodes2]);

        eai[0] = 0;
        for (EdgeIndex i = 0; i < numEdges2; i++) {
            eai[i + 1] = hypergraph.edgeSize(i) + eai[i];
            ewt[i] = hypergraph.edgeWeight(i);
            if (hypergraph.edgeWeight(i) != 1) hasWeightedEdges = true;
        }

        for (EdgeID edgeID : hypergraph.edges()) {
            NodeIndex numPinsAdded = 0;
            for (NodeID pinID : hypergraph.pins(edgeID)) {
                eal[eai[edgeID] + numPinsAdded++] = pinID;
                nwt[pinID] = hypergraph.nodeWeight(pinID);
            }
        }

        mt_kahypar_hypergraph_t dyn_wrapper = mt_kahypar_create_hypergraph(
            HIGHEST_QUALITY, numNodes2, numEdges2, eai.get(), eal.get(), ewt.get(), nwt.get());
        DynamicHypergraph& dynHG = mt_kahypar::utils::cast<DynamicHypergraph>(dyn_wrapper);
        totalComputingTime += t.elapsed();

        t.restart();
        SubmodularMincut solver(numNodes2, numEdges2, orderingType, OrderingMode::SINGLE, hasWeightedEdges, 1);
        SubmodularMincutResult result = solver.solve(dynHG);
        minEdgeCut = std::min(minEdgeCut, result.minEdgeCut);
        totalComputingTime += t.elapsed();

        mt_kahypar_free_hypergraph(dyn_wrapper);
    }

    if (get_num_edges() == 0 && get_num_nodes() > 1)
        minEdgeCut = 0;

    uint32_t cut_value = minEdgeCut;
    double total_time = totalComputingTime;
#endif

    mt_kahypar_free_hypergraph(hg_wrapper);
    return std::make_tuple(cut_value, total_time);
}


// ============================================================================
// submodular: direct submodular mincut solver
// ============================================================================

static std::tuple<uint32_t, double> py_submodular(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes,
    const std::string& ordering_type_str,
    int seed,
    int num_threads)
{
    ensure_thread_pool(static_cast<size_t>(num_threads));
    OutputSuppressor suppress;

    RandomFunctions::set_seed(seed);

    mt_kahypar_hypergraph_t hg_wrapper = build_dynamic_hypergraph(eptr, everts, edge_weights, num_nodes);
    DynamicHypergraph& hypergraph = mt_kahypar::utils::cast<DynamicHypergraph>(hg_wrapper);

    OrderingType orderingType = parse_ordering_type(ordering_type_str);

    EdgeIndex inputNumEdges = hypergraph.initialNumEdges() - hypergraph.numRemovedHyperedges();

    // Determine if the hypergraph has weighted edges
    bool hasWeightedEdges = false;
    for (EdgeID edgeID : hypergraph.edges())
        if (hypergraph.edgeIsEnabled(edgeID) && hypergraph.edgeWeight(edgeID) != 1) {
            hasWeightedEdges = true;
            break;
        }

    timer t;
    double totalComputingTime = 0;

    t.restart();
    SubmodularMincut solver(hypergraph.initialNumNodes(), inputNumEdges, orderingType, OrderingMode::SINGLE, hasWeightedEdges, 1);
    totalComputingTime += t.elapsed();

    t.restart();
    SubmodularMincutResult result = solver.solve(hypergraph);
    totalComputingTime += t.elapsed();

    uint32_t cut_value = result.minEdgeCut;
    mt_kahypar_free_hypergraph(hg_wrapper);

    return std::make_tuple(cut_value, totalComputingTime);
}


// ============================================================================
// trimmer: k-trimmer certificate + submodular solver
// ============================================================================

static std::tuple<uint32_t, double> py_trimmer(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    int num_nodes,
    const std::string& ordering_type_str,
    int seed,
    int num_threads)
{
    ensure_thread_pool(static_cast<size_t>(num_threads));
    OutputSuppressor suppress;

    RandomFunctions::set_seed(seed);

    // Build an unweighted static hypergraph (trimmer only works unweighted)
    py::array_t<int32_t> empty_weights(0);
    mt_kahypar_hypergraph_t hg_wrapper = build_static_hypergraph(eptr, everts, empty_weights, num_nodes);
    StaticHypergraph& hypergraph = mt_kahypar::utils::cast<StaticHypergraph>(hg_wrapper);

    OrderingType orderingType = parse_ordering_type(ordering_type_str);

    EdgeIndex inputNumEdges = hypergraph.initialNumEdges() - hypergraph.numRemovedHyperedges();
    NodeIndex inputNumNodes = hypergraph.initialNumNodes() - hypergraph.numRemovedHypernodes();

    // Force all edges to weight 1 (unweighted)
    for (EdgeID edgeID : hypergraph.edges())
        if (hypergraph.edgeIsEnabled(edgeID))
            hypergraph.setEdgeWeight(edgeID, 1);

    bool hasWeightedEdges = false;

    timer t;
    double totalComputingTime = 0;

    // Preprocessing: compute MA-ordering
    t.restart();
    std::vector<NodeID> nodeOrdering(inputNumNodes);
    std::vector<EdgeID> edgeHeadOrdering(inputNumEdges);
    std::vector<NodeID> edgeHead(inputNumEdges);

    Orderer<StaticHypergraph, EdgeWeight> inputOrderer(
        hypergraph.initialNumNodes(), inputNumEdges, OrderingType::MA,
        hasWeightedEdges, RandomFunctions::get_random_engine());
    inputOrderer.compute_ordering(hypergraph, inputNumNodes, &nodeOrdering, &edgeHeadOrdering, &edgeHead);

    // Initialize the trimmer for dynamic hypergraph output
    Trimmer trimmer(hypergraph.initialNumNodes(), inputNumNodes, inputNumEdges,
                    hypergraph, nodeOrdering, edgeHeadOrdering, edgeHead, HIGHEST_QUALITY);
    trimmer.build_backward_edges();

    SubmodularMincut mincutSolver(hypergraph.initialNumNodes(), inputNumEdges,
                                  orderingType, OrderingMode::SINGLE, hasWeightedEdges, 1);

    totalComputingTime += t.elapsed();

    // Exponential search on k starting at 2
    TrimmerValue k = 2;
    CutValue minEdgeCut = 0;

    while (true) {
        t.restart();

        mt_kahypar_hypergraph_t trimmed_wrapper = trimmer.create_k_trimmed_certificate(k);
        DynamicHypergraph& trimmedHG = mt_kahypar::utils::cast<DynamicHypergraph>(trimmed_wrapper);

        SubmodularMincutResult result = mincutSolver.solve(trimmedHG);
        minEdgeCut = result.minEdgeCut;

        mt_kahypar_free_hypergraph(trimmed_wrapper);
        totalComputingTime += t.elapsed();

        if (k > minEdgeCut)
            break;

        k *= 2;
    }

    mt_kahypar_free_hypergraph(hg_wrapper);
    return std::make_tuple(static_cast<uint32_t>(minEdgeCut), totalComputingTime);
}


// ============================================================================
// ILP solver (only when Gurobi is available)
// ============================================================================
#ifdef HEICUT_HAS_GUROBI
static std::tuple<uint32_t, double, bool> py_ilp(
    py::array_t<int64_t, py::array::c_style> eptr,
    py::array_t<int32_t, py::array::c_style> everts,
    py::array_t<int32_t, py::array::c_style> edge_weights,
    int num_nodes,
    double ilp_timeout,
    int seed,
    int num_threads)
{
    ensure_thread_pool(static_cast<size_t>(num_threads));
    OutputSuppressor suppress;

    RandomFunctions::set_seed(seed);

    mt_kahypar_hypergraph_t hg_wrapper = build_static_hypergraph(eptr, everts, edge_weights, num_nodes);
    StaticHypergraph& hypergraph = mt_kahypar::utils::cast<StaticHypergraph>(hg_wrapper);

    timer t;
    bool is_optimal = false;

    try {
        NodeIndex currentNodes = hypergraph.initialNumNodes() - hypergraph.numRemovedHypernodes();
        EdgeIndex currentEdges = hypergraph.initialNumEdges() - hypergraph.numRemovedHyperedges();

        GRBEnv env = GRBEnv(true);
        env.set(GRB_IntParam_OutputFlag, 0); // Suppress Gurobi output
        env.start();

        ILPMincut solver(currentNodes, currentEdges, env, seed, ilp_timeout,
                         ILPMode::BIP, static_cast<size_t>(num_threads));
        solver.add_node_variables_and_constraints(hypergraph);
        solver.add_edge_variables_and_constraints(hypergraph);
        solver.set_objective();
        solver.optimize();
        CutValue result = solver.get_result(hypergraph);

        double elapsed = t.elapsed();
        is_optimal = true;

        mt_kahypar_free_hypergraph(hg_wrapper);
        return std::make_tuple(static_cast<uint32_t>(result), elapsed, is_optimal);

    } catch (GRBException& e) {
        double elapsed = t.elapsed();
        mt_kahypar_free_hypergraph(hg_wrapper);
        throw std::runtime_error("Gurobi error " + std::to_string(e.getErrorCode()) + ": " + e.getMessage());
    }
}
#endif


// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(_heicut, m) {
    m.doc() = "Python bindings for HeiCut hypergraph minimum cut solvers";

#ifdef HEICUT_HAS_GUROBI
    m.attr("HAS_GUROBI") = true;
#else
    m.attr("HAS_GUROBI") = false;
#endif

    m.def("kernelizer", &py_kernelizer,
          py::arg("eptr"), py::arg("everts"), py::arg("edge_weights"),
          py::arg("num_nodes"),
          py::arg("ordering_type"), py::arg("lp_iterations"),
          py::arg("seed"), py::arg("num_threads"),
          R"doc(
          Compute hypergraph minimum cut using kernelization + submodular solver.

          Applies pruning rules and label propagation to reduce the hypergraph,
          then solves the remaining kernel with the submodular mincut algorithm.

          Parameters
          ----------
          eptr : ndarray[int64]
              Edge pointer array (length num_edges + 1).
          everts : ndarray[int32]
              Concatenated vertex lists per edge.
          edge_weights : ndarray[int32]
              Edge weight array (length num_edges). Empty for unweighted.
          num_nodes : int
              Number of vertices in the hypergraph.
          ordering_type : str
              Node ordering: 'ma', 'tight', or 'queyranne'.
          lp_iterations : int
              Number of label propagation iterations (0 to disable).
          seed : int
              Random seed.
          num_threads : int
              Number of threads for mt-kahypar initialization.

          Returns
          -------
          tuple[int, float]
              (cut_value, computation_time_seconds).
          )doc");

    m.def("submodular", &py_submodular,
          py::arg("eptr"), py::arg("everts"), py::arg("edge_weights"),
          py::arg("num_nodes"),
          py::arg("ordering_type"),
          py::arg("seed"), py::arg("num_threads"),
          R"doc(
          Compute hypergraph minimum cut via submodular optimization.

          Directly solves the minimum cut problem using the Queyranne/MA/Tight
          ordering-based contraction algorithm on a dynamic hypergraph.

          Parameters
          ----------
          eptr : ndarray[int64]
              Edge pointer array (length num_edges + 1).
          everts : ndarray[int32]
              Concatenated vertex lists per edge.
          edge_weights : ndarray[int32]
              Edge weight array (length num_edges). Empty for unweighted.
          num_nodes : int
              Number of vertices in the hypergraph.
          ordering_type : str
              Node ordering: 'ma', 'tight', or 'queyranne'.
          seed : int
              Random seed.
          num_threads : int
              Number of threads for mt-kahypar initialization.

          Returns
          -------
          tuple[int, float]
              (cut_value, computation_time_seconds).
          )doc");

    m.def("trimmer", &py_trimmer,
          py::arg("eptr"), py::arg("everts"),
          py::arg("num_nodes"),
          py::arg("ordering_type"),
          py::arg("seed"), py::arg("num_threads"),
          R"doc(
          Compute exact hypergraph minimum cut using the k-trimmer of Chekuri and Xu.

          Builds a k-trimmed certificate and solves it with the submodular solver,
          doubling k until k > mincut (guaranteeing exactness). Unweighted only.

          Parameters
          ----------
          eptr : ndarray[int64]
              Edge pointer array (length num_edges + 1).
          everts : ndarray[int32]
              Concatenated vertex lists per edge.
          num_nodes : int
              Number of vertices in the hypergraph.
          ordering_type : str
              Node ordering: 'ma', 'tight', or 'queyranne'.
          seed : int
              Random seed.
          num_threads : int
              Number of threads for mt-kahypar initialization.

          Returns
          -------
          tuple[int, float]
              (cut_value, computation_time_seconds).
          )doc");

#ifdef HEICUT_HAS_GUROBI
    m.def("ilp", &py_ilp,
          py::arg("eptr"), py::arg("everts"), py::arg("edge_weights"),
          py::arg("num_nodes"),
          py::arg("ilp_timeout"),
          py::arg("seed"), py::arg("num_threads"),
          R"doc(
          Compute hypergraph minimum cut using ILP (requires Gurobi).

          Formulates the hypergraph minimum cut problem as a binary integer
          program and solves it with Gurobi.

          Parameters
          ----------
          eptr : ndarray[int64]
              Edge pointer array (length num_edges + 1).
          everts : ndarray[int32]
              Concatenated vertex lists per edge.
          edge_weights : ndarray[int32]
              Edge weight array (length num_edges). Empty for unweighted.
          num_nodes : int
              Number of vertices in the hypergraph.
          ilp_timeout : float
              Timeout in seconds for the ILP solver.
          seed : int
              Random seed.
          num_threads : int
              Number of threads.

          Returns
          -------
          tuple[int, float, bool]
              (cut_value, computation_time_seconds, is_optimal).
          )doc");
#endif
}
