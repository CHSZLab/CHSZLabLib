"""Tests for hypergraph b-matching (static + streaming)."""

import numpy as np
import pytest

from chszlablib import (
    HyperGraph,
    IndependenceProblems,
    BMatchingResult,
    StreamingBMatcher,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def make_small_hypergraph():
    """5 nodes, 4 edges: {0,1}, {1,2}, {2,3}, {3,4}  weights 5,3,7,2."""
    return HyperGraph.from_edge_list(
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        num_nodes=5,
        edge_weights=[5, 3, 7, 2],
    )


def make_triangle_hypergraph():
    """4 nodes, 3 edges: {0,1,2}, {1,2,3}, {0,3}  weights 10,8,4."""
    return HyperGraph.from_edge_list(
        [[0, 1, 2], [1, 2, 3], [0, 3]],
        num_nodes=4,
        edge_weights=[10, 8, 4],
    )


# ── Static B-Matching Tests ──────────────────────────────────────────────

class TestStaticBMatching:
    def test_basic_greedy(self):
        hg = make_small_hypergraph()
        r = IndependenceProblems.bmatching(hg, algorithm="greedy_weight_desc")
        assert isinstance(r, BMatchingResult)
        assert r.num_matched > 0
        assert r.total_weight > 0
        assert len(r.matched_edges) == r.num_matched

    def test_capacity_default_one(self):
        """With default capacity=1, no node appears in >1 matched edge."""
        hg = make_small_hypergraph()
        r = IndependenceProblems.bmatching(hg, algorithm="greedy_weight_desc")
        hg.finalize()
        node_counts = np.zeros(hg.num_nodes, dtype=int)
        for eidx in r.matched_edges:
            start = int(hg.eptr[eidx])
            end = int(hg.eptr[eidx + 1])
            for v in hg.everts[start:end]:
                node_counts[v] += 1
        assert np.all(node_counts <= 1)

    def test_capacity_two(self):
        """With capacity=2, nodes can appear in up to 2 matched edges."""
        hg = make_small_hypergraph()
        hg._finalized = False  # Reset to allow setting capacities
        # Rebuild to set capacities
        hg2 = HyperGraph(5, 4)
        hg2.set_edge(0, [0, 1])
        hg2.set_edge(1, [1, 2])
        hg2.set_edge(2, [2, 3])
        hg2.set_edge(3, [3, 4])
        hg2.set_edge_weight(0, 5)
        hg2.set_edge_weight(1, 3)
        hg2.set_edge_weight(2, 7)
        hg2.set_edge_weight(3, 2)
        hg2.set_capacities(np.array([2, 2, 2, 2, 2]))
        r = IndependenceProblems.bmatching(hg2, algorithm="greedy_weight_desc")
        # With capacity 2, we can match more edges
        assert r.num_matched >= 2

    @pytest.mark.parametrize("algorithm", [
        "greedy_random",
        "greedy_weight_desc",
        "greedy_weight_asc",
        "greedy_degree_asc",
        "greedy_degree_desc",
        "greedy_weight_degree_ratio_desc",
        "greedy_weight_degree_ratio_asc",
        "reductions",
        "ils",
    ])
    def test_all_algorithms(self, algorithm):
        hg = make_small_hypergraph()
        r = IndependenceProblems.bmatching(hg, algorithm=algorithm, seed=42)
        assert isinstance(r, BMatchingResult)
        assert r.num_matched >= 0
        assert r.total_weight >= 0

    def test_invalid_algorithm(self):
        hg = make_small_hypergraph()
        with pytest.raises(Exception):
            IndependenceProblems.bmatching(hg, algorithm="nonexistent")

    def test_hyperedge_matching(self):
        """Test with hyperedges of size > 2."""
        hg = make_triangle_hypergraph()
        r = IndependenceProblems.bmatching(hg, algorithm="greedy_weight_desc")
        assert r.num_matched >= 1
        # The best single-edge matching is edge 0 (weight 10)
        assert r.total_weight >= 10

    def test_single_edge(self):
        hg = HyperGraph.from_edge_list([[0, 1]], num_nodes=2, edge_weights=[42])
        r = IndependenceProblems.bmatching(hg)
        assert r.num_matched == 1
        assert r.total_weight == 42.0

    def test_seed_reproducibility(self):
        hg = make_small_hypergraph()
        r1 = IndependenceProblems.bmatching(hg, algorithm="greedy_random", seed=123)
        # Rebuild because finalize is idempotent
        hg2 = make_small_hypergraph()
        r2 = IndependenceProblems.bmatching(hg2, algorithm="greedy_random", seed=123)
        assert r1.total_weight == r2.total_weight
        np.testing.assert_array_equal(sorted(r1.matched_edges), sorted(r2.matched_edges))


# ── Streaming B-Matching Tests ───────────────────────────────────────────

class TestStreamingBMatcher:
    def test_basic_streaming(self):
        sm = StreamingBMatcher(5, algorithm="greedy")
        sm.add_edge([0, 1], 5.0)
        sm.add_edge([1, 2], 3.0)
        sm.add_edge([2, 3], 7.0)
        sm.add_edge([3, 4], 2.0)
        r = sm.finish()
        assert isinstance(r, BMatchingResult)
        assert r.num_matched > 0
        assert r.total_weight > 0

    @pytest.mark.parametrize("algorithm", StreamingBMatcher.ALGORITHMS)
    def test_all_streaming_algorithms(self, algorithm):
        sm = StreamingBMatcher(5, algorithm=algorithm)
        sm.add_edge([0, 1], 5.0)
        sm.add_edge([1, 2], 3.0)
        sm.add_edge([2, 3], 7.0)
        sm.add_edge([3, 4], 2.0)
        r = sm.finish()
        assert isinstance(r, BMatchingResult)
        assert r.num_matched >= 1
        assert r.total_weight > 0

    def test_streaming_reset(self):
        sm = StreamingBMatcher(5, algorithm="greedy")
        sm.add_edge([0, 1], 5.0)
        sm.add_edge([2, 3], 7.0)
        r1 = sm.finish()

        sm.reset()
        sm.add_edge([0, 1], 5.0)
        sm.add_edge([2, 3], 7.0)
        r2 = sm.finish()

        assert r1.total_weight == r2.total_weight
        assert r1.num_matched == r2.num_matched

    def test_streaming_edge_count(self):
        sm = StreamingBMatcher(4)
        assert sm.num_edges_streamed == 0
        sm.add_edge([0, 1], 1.0)
        assert sm.num_edges_streamed == 1
        sm.add_edge([2, 3], 1.0)
        assert sm.num_edges_streamed == 2

    def test_streaming_repr(self):
        sm = StreamingBMatcher(10, algorithm="naive")
        assert "StreamingBMatcher" in repr(sm)
        assert "naive" in repr(sm)

    def test_streaming_single_edge(self):
        sm = StreamingBMatcher(2)
        sm.add_edge([0, 1], 42.0)
        r = sm.finish()
        assert r.num_matched == 1
        assert r.total_weight == 42.0

    def test_streaming_hyperedges(self):
        """Test with edges of size > 2."""
        sm = StreamingBMatcher(4, algorithm="greedy")
        sm.add_edge([0, 1, 2], 10.0)
        sm.add_edge([1, 2, 3], 8.0)
        sm.add_edge([0, 3], 4.0)
        r = sm.finish()
        assert r.num_matched >= 1

    def test_invalid_algorithm(self):
        with pytest.raises(Exception):
            StreamingBMatcher(5, algorithm="nonexistent")

    def test_default_algorithm(self):
        assert StreamingBMatcher.DEFAULT_ALGORITHM == "greedy"

    def test_disconnected_edges(self):
        """Completely disjoint edges should all be matched."""
        sm = StreamingBMatcher(6, algorithm="naive")
        sm.add_edge([0, 1], 1.0)
        sm.add_edge([2, 3], 1.0)
        sm.add_edge([4, 5], 1.0)
        r = sm.finish()
        assert r.num_matched == 3
        assert r.total_weight == 3.0


# ── HyperGraph Capacity Tests ───────────────────────────────────────────

class TestHyperGraphCapacity:
    def test_default_capacity(self):
        hg = HyperGraph.from_edge_list([[0, 1]], num_nodes=2)
        assert np.all(hg.capacities == 1)

    def test_set_capacity(self):
        hg = HyperGraph(3, 1)
        hg.set_edge(0, [0, 1, 2])
        hg.set_capacity(0, 3)
        hg.set_capacity(1, 2)
        hg.finalize()
        assert hg.capacities[0] == 3
        assert hg.capacities[1] == 2
        assert hg.capacities[2] == 1  # default

    def test_set_capacities_array(self):
        hg = HyperGraph(4, 1)
        hg.set_edge(0, [0, 1, 2, 3])
        hg.set_capacities([2, 3, 1, 4])
        hg.finalize()
        np.testing.assert_array_equal(hg.capacities, [2, 3, 1, 4])

    def test_capacity_validation(self):
        hg = HyperGraph(2, 1)
        hg.set_edge(0, [0, 1])
        with pytest.raises(Exception):
            hg.set_capacity(0, 0)  # capacity < 1

    def test_capacity_after_finalize(self):
        hg = HyperGraph.from_edge_list([[0, 1]], num_nodes=2)
        with pytest.raises(Exception):
            hg.set_capacity(0, 2)
