#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Reimplementation of the 5 streaming matching algorithms from HeiHGM/Streaming.
// The original code in greedy.cc is deeply coupled to Abseil/Protobuf via the
// AlgorithmImpl factory layer. We reimplement the core streaming logic here
// directly, which is simpler and avoids those dependencies entirely.

namespace streaming_matching {

struct Hyperedge {
    std::vector<size_t> pins;
    double weight;
};

// =========================================================================
// Algorithm 1: Naive — simple binary matched flag per node
// =========================================================================
struct NaiveState {
    std::vector<bool> matched_at;
    std::vector<size_t> matched_edges;
    double total_weight = 0.0;

    explicit NaiveState(size_t num_nodes)
        : matched_at(num_nodes, false) {}

    void stream(size_t edge_idx, const Hyperedge& e) {
        for (auto p : e.pins) {
            if (matched_at[p]) return;
        }
        // All pins free — match this edge
        matched_edges.push_back(edge_idx);
        total_weight += e.weight;
        for (auto p : e.pins) {
            matched_at[p] = true;
        }
    }
};

// =========================================================================
// Algorithm 2: Greedy — phi-based weight tracking with epsilon
// =========================================================================
struct GreedyState {
    std::vector<double> phi;
    std::vector<Hyperedge> stack;
    std::vector<size_t> stack_indices;
    double eps;

    explicit GreedyState(size_t num_nodes, double epsilon)
        : phi(num_nodes, 0.0), eps(epsilon) {}

    void stream(size_t edge_idx, const Hyperedge& e) {
        double weight = 0.0;
        for (auto p : e.pins) {
            weight += phi[p];
        }
        if (e.weight > weight * (1.0 + eps)) {
            double offset = e.weight - weight;
            stack.push_back(e);
            stack_indices.push_back(edge_idx);
            for (auto p : e.pins) {
                phi[p] += offset;
            }
        }
    }

    // Transform: iterate stack in reverse, greedily match
    std::pair<std::vector<size_t>, double> finalize(size_t num_nodes) {
        std::vector<bool> matched_at(num_nodes, false);
        std::vector<size_t> result;
        double total_weight = 0.0;
        for (int i = static_cast<int>(stack.size()) - 1; i >= 0; i--) {
            bool matchable = true;
            for (auto p : stack[i].pins) {
                if (matched_at[p]) { matchable = false; break; }
            }
            if (matchable) {
                result.push_back(stack_indices[i]);
                total_weight += stack[i].weight;
                for (auto p : stack[i].pins) {
                    matched_at[p] = true;
                }
            }
        }
        return {result, total_weight};
    }
};

// =========================================================================
// Algorithm 3: GreedySet — list-based with per-vertex shortcut pointers
// =========================================================================
struct GreedySetState {
    // Per-vertex: pointer index to the edge currently covering this vertex
    // -1 means no edge
    std::vector<int> short_cut;
    std::vector<Hyperedge> edges;
    std::vector<size_t> edge_indices;
    std::vector<bool> alive;  // whether each stored edge is still active
    double eps;

    explicit GreedySetState(size_t num_nodes, double epsilon)
        : short_cut(num_nodes, -1), eps(epsilon) {}

    void stream(size_t edge_idx, const Hyperedge& e) {
        double weight = 0.0;
        for (auto p : e.pins) {
            if (short_cut[p] >= 0 && alive[short_cut[p]]) {
                weight += edges[short_cut[p]].weight;
            }
        }
        if (e.weight > weight * (1.0 + eps)) {
            // Evict conflicting edges
            for (auto p : e.pins) {
                if (short_cut[p] >= 0 && alive[short_cut[p]]) {
                    int old_idx = short_cut[p];
                    // Remove old edge's shortcuts
                    for (auto q : edges[old_idx].pins) {
                        short_cut[q] = -1;
                    }
                    alive[old_idx] = false;
                }
            }
            // Add new edge
            int store_idx = static_cast<int>(edges.size());
            edges.push_back(e);
            edge_indices.push_back(edge_idx);
            alive.push_back(true);
            for (auto p : e.pins) {
                short_cut[p] = store_idx;
            }
        }
    }

    std::pair<std::vector<size_t>, double> finalize() {
        std::vector<size_t> result;
        double total_weight = 0.0;
        // Collect alive edges (deduplicate via short_cut)
        std::vector<bool> seen(edges.size(), false);
        for (size_t v = 0; v < short_cut.size(); v++) {
            if (short_cut[v] >= 0 && alive[short_cut[v]] && !seen[short_cut[v]]) {
                seen[short_cut[v]] = true;
                result.push_back(edge_indices[short_cut[v]]);
                total_weight += edges[short_cut[v]].weight;
            }
        }
        return {result, total_weight};
    }
};

// =========================================================================
// Algorithm 4: BestEvict — greedy_set with data-dependent epsilon
//   Pre-streams to compute optimal epsilon, then does a full stream pass.
// =========================================================================
struct BestEvictState {
    size_t num_nodes;
    std::vector<Hyperedge> all_edges;
    std::vector<size_t> all_indices;

    explicit BestEvictState(size_t n) : num_nodes(n) {}

    void stream(size_t edge_idx, const Hyperedge& e) {
        all_edges.push_back(e);
        all_indices.push_back(edge_idx);
    }

    std::pair<std::vector<size_t>, double> finalize() {
        // Pre-stream with eps=0 to find optimal eps
        double best_weight = 0.0;
        double best_eps = 0.0;

        // Try a few epsilon values
        for (double test_eps : {0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}) {
            GreedySetState gs(num_nodes, test_eps);
            for (size_t i = 0; i < all_edges.size(); i++) {
                gs.stream(all_indices[i], all_edges[i]);
            }
            auto [edges, w] = gs.finalize();
            if (w > best_weight) {
                best_weight = w;
                best_eps = test_eps;
            }
        }

        // Run with best eps
        GreedySetState gs(num_nodes, best_eps);
        for (size_t i = 0; i < all_edges.size(); i++) {
            gs.stream(all_indices[i], all_edges[i]);
        }
        return gs.finalize();
    }
};

// =========================================================================
// Algorithm 5: Lenient — half-scaled phi distribution
// =========================================================================
struct LenientState {
    std::vector<double> phi;
    std::vector<Hyperedge> stack;
    std::vector<size_t> stack_indices;
    double eps;

    explicit LenientState(size_t num_nodes, double epsilon)
        : phi(num_nodes, 0.0), eps(epsilon) {}

    void stream(size_t edge_idx, const Hyperedge& e) {
        double weight = 0.0;
        for (auto p : e.pins) {
            weight += phi[p];
        }
        if (e.weight > weight * (1.0 + eps)) {
            // Lenient: distribute weight / |pins| to each pin
            double per_pin = e.weight / static_cast<double>(e.pins.size());
            stack.push_back(e);
            stack_indices.push_back(edge_idx);
            for (auto p : e.pins) {
                phi[p] += per_pin;
            }
        }
    }

    std::pair<std::vector<size_t>, double> finalize(size_t num_nodes) {
        std::vector<bool> matched_at(num_nodes, false);
        std::vector<size_t> result;
        double total_weight = 0.0;
        for (int i = static_cast<int>(stack.size()) - 1; i >= 0; i--) {
            bool matchable = true;
            for (auto p : stack[i].pins) {
                if (matched_at[p]) { matchable = false; break; }
            }
            if (matchable) {
                result.push_back(stack_indices[i]);
                total_weight += stack[i].weight;
                for (auto p : stack[i].pins) {
                    matched_at[p] = true;
                }
            }
        }
        return {result, total_weight};
    }
};

} // namespace streaming_matching


// pybind11 class wrapping the streaming matcher
class StreamingMatcher {
public:
    StreamingMatcher(size_t num_nodes, const std::string& algorithm,
                     py::array_t<int32_t, py::array::c_style> capacities,
                     int seed, double epsilon)
        : num_nodes_(num_nodes), algorithm_(algorithm), epsilon_(epsilon)
    {
        (void)seed; // seed reserved for future shuffled-order variants
        auto cap = capacities.unchecked<1>();
        capacities_.resize(num_nodes);
        for (size_t i = 0; i < num_nodes; i++) {
            capacities_[i] = static_cast<int>(cap(i));
        }
        init_state();
    }

    void add_edge(const std::vector<int>& nodes, double weight) {
        streaming_matching::Hyperedge e;
        e.pins.reserve(nodes.size());
        for (auto n : nodes) {
            e.pins.push_back(static_cast<size_t>(n));
        }
        e.weight = weight;

        size_t idx = edges_.size();
        edges_.push_back(e);

        // For naive, stream immediately
        if (algorithm_ == "naive") {
            naive_state_->stream(idx, e);
        } else if (algorithm_ == "greedy") {
            greedy_state_->stream(idx, e);
        } else if (algorithm_ == "greedy_set") {
            greedy_set_state_->stream(idx, e);
        } else if (algorithm_ == "best_evict") {
            best_evict_state_->stream(idx, e);
        } else if (algorithm_ == "lenient") {
            lenient_state_->stream(idx, e);
        }
    }

    // Finalize and return (matched_edge_indices, total_weight)
    std::tuple<py::array_t<int32_t>, double> finish() {
        std::vector<size_t> matched;
        double total_weight = 0.0;

        if (algorithm_ == "naive") {
            matched = naive_state_->matched_edges;
            total_weight = naive_state_->total_weight;
        } else if (algorithm_ == "greedy") {
            auto [m, w] = greedy_state_->finalize(num_nodes_);
            matched = m;
            total_weight = w;
        } else if (algorithm_ == "greedy_set") {
            auto [m, w] = greedy_set_state_->finalize();
            matched = m;
            total_weight = w;
        } else if (algorithm_ == "best_evict") {
            auto [m, w] = best_evict_state_->finalize();
            matched = m;
            total_weight = w;
        } else if (algorithm_ == "lenient") {
            auto [m, w] = lenient_state_->finalize(num_nodes_);
            matched = m;
            total_weight = w;
        }

        py::array_t<int32_t> result(matched.size());
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < matched.size(); i++) {
            r(i) = static_cast<int32_t>(matched[i]);
        }
        return std::make_tuple(result, total_weight);
    }

    void reset() {
        edges_.clear();
        init_state();
    }

private:
    void init_state() {
        naive_state_.reset();
        greedy_state_.reset();
        greedy_set_state_.reset();
        best_evict_state_.reset();
        lenient_state_.reset();

        if (algorithm_ == "naive") {
            naive_state_ = std::make_unique<streaming_matching::NaiveState>(num_nodes_);
        } else if (algorithm_ == "greedy") {
            greedy_state_ = std::make_unique<streaming_matching::GreedyState>(num_nodes_, epsilon_);
        } else if (algorithm_ == "greedy_set") {
            greedy_set_state_ = std::make_unique<streaming_matching::GreedySetState>(num_nodes_, epsilon_);
        } else if (algorithm_ == "best_evict") {
            best_evict_state_ = std::make_unique<streaming_matching::BestEvictState>(num_nodes_);
        } else if (algorithm_ == "lenient") {
            lenient_state_ = std::make_unique<streaming_matching::LenientState>(num_nodes_, epsilon_);
        } else {
            throw std::invalid_argument("Unknown streaming algorithm: " + algorithm_);
        }
    }

    size_t num_nodes_;
    std::string algorithm_;
    double epsilon_;
    std::vector<int> capacities_;
    std::vector<streaming_matching::Hyperedge> edges_;

    std::unique_ptr<streaming_matching::NaiveState> naive_state_;
    std::unique_ptr<streaming_matching::GreedyState> greedy_state_;
    std::unique_ptr<streaming_matching::GreedySetState> greedy_set_state_;
    std::unique_ptr<streaming_matching::BestEvictState> best_evict_state_;
    std::unique_ptr<streaming_matching::LenientState> lenient_state_;
};


PYBIND11_MODULE(_streaming_bmatching, m) {
    m.doc() = "Python bindings for streaming hypergraph matching (HeiHGM)";

    py::class_<StreamingMatcher>(m, "StreamingMatcher")
        .def(py::init<size_t, const std::string&,
                       py::array_t<int32_t, py::array::c_style>,
                       int, double>(),
             py::arg("num_nodes"), py::arg("algorithm"),
             py::arg("capacities"), py::arg("seed"), py::arg("epsilon"),
             R"doc(
             Create a streaming matcher.

             Parameters
             ----------
             num_nodes : int
                 Number of vertices.
             algorithm : str
                 One of "naive", "greedy", "greedy_set", "best_evict", "lenient".
             capacities : ndarray[int32]
                 Node capacity array (length num_nodes).
             seed : int
                 Random seed.
             epsilon : float
                 Epsilon parameter for greedy comparisons.
             )doc")
        .def("add_edge", &StreamingMatcher::add_edge,
             py::arg("nodes"), py::arg("weight"),
             "Feed one hyperedge (list of node IDs, weight) to the streamer.")
        .def("finish", &StreamingMatcher::finish,
             "Finalize and return (matched_edge_indices, total_weight).")
        .def("reset", &StreamingMatcher::reset,
             "Reset state for re-streaming.");
}
