"""Tests for b-matching (Bmatching)."""

import numpy as np
import pytest

from chszlablib import Graph, bmatching, hypergraph_bmatching


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_bmatching(g, result, capacities=1):
    """Check that the b-matching is valid: no node exceeds its capacity."""
    g.finalize()
    n = g.num_nodes

    if isinstance(capacities, int):
        cap = np.full(n, capacities, dtype=np.int32)
    else:
        cap = np.asarray(capacities, dtype=np.int32)

    # Build unique edge list (u < v) in CSR order
    edges = []
    for u in range(n):
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            v = g.adjncy[idx]
            if u < v:
                edges.append((u, v))

    # Count incidences per node
    node_count = np.zeros(n, dtype=np.int32)
    for edge_idx in result.matched_edges:
        u, v = edges[edge_idx]
        node_count[u] += 1
        node_count[v] += 1

    for i in range(n):
        assert node_count[i] <= cap[i], (
            f"Node {i}: matched {node_count[i]} edges, capacity {cap[i]}"
        )


def validate_hypergraph_bmatching(num_nodes, edges, result, capacities=1):
    """Check validity for a hypergraph b-matching."""
    if isinstance(capacities, int):
        cap = np.full(num_nodes, capacities, dtype=np.int32)
    else:
        cap = np.asarray(capacities, dtype=np.int32)

    node_count = np.zeros(num_nodes, dtype=np.int32)
    for edge_idx in result.matched_edges:
        for node in edges[edge_idx]:
            node_count[node] += 1

    for i in range(num_nodes):
        assert node_count[i] <= cap[i], (
            f"Node {i}: matched {node_count[i]} edges, capacity {cap[i]}"
        )


# ---------------------------------------------------------------------------
# Standard graph tests
# ---------------------------------------------------------------------------

class TestBMatchingBasic:
    """Basic b-matching tests on standard graphs."""

    def test_single_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        r = bmatching(g, capacities=1, algorithm="greedy")
        assert r.size == 1
        assert r.weight == 1
        validate_bmatching(g, r, capacities=1)

    def test_path_graph(self):
        """Path 0-1-2-3: b=1 matching can have at most 2 edges."""
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        r = bmatching(g, capacities=1, algorithm="greedy")
        assert r.size <= 2
        validate_bmatching(g, r, capacities=1)

    def test_triangle(self):
        """Triangle: b=1 matching can have at most 1 edge."""
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        r = bmatching(g, capacities=1, algorithm="greedy")
        assert r.size == 1
        validate_bmatching(g, r, capacities=1)

    def test_complete_k4(self):
        """K4: b=1, maximum matching = 2."""
        g = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(i, j)
        r = bmatching(g, capacities=1, algorithm="ils", time_limit=1.0)
        assert r.size == 2
        validate_bmatching(g, r, capacities=1)

    def test_star_graph(self):
        """Star: center=0, 4 leaves. b=1: exactly 1 edge matched."""
        g = Graph(5)
        for i in range(1, 5):
            g.add_edge(0, i)
        r = bmatching(g, capacities=1, algorithm="greedy")
        assert r.size == 1
        validate_bmatching(g, r, capacities=1)

    def test_weighted_edges(self):
        """Greedy picks heaviest feasible edges."""
        g = Graph(4)
        g.add_edge(0, 1, weight=10)
        g.add_edge(1, 2, weight=1)
        g.add_edge(2, 3, weight=10)
        r = bmatching(g, capacities=1, algorithm="greedy")
        assert r.weight >= 10
        validate_bmatching(g, r, capacities=1)

    def test_non_uniform_capacities(self):
        """Nodes with higher capacity can be in more edges."""
        # Star: center=0, 4 leaves. center capacity=3.
        g = Graph(5)
        for i in range(1, 5):
            g.add_edge(0, i)
        cap = np.array([3, 1, 1, 1, 1], dtype=np.int32)
        r = bmatching(g, capacities=cap, algorithm="ils", time_limit=1.0)
        assert r.size == 3
        validate_bmatching(g, r, capacities=cap)

    def test_empty_graph(self):
        g = Graph(5)
        r = bmatching(g, capacities=1, algorithm="greedy")
        assert r.size == 0
        assert r.weight == 0


class TestBMatchingAlgorithms:
    """Test all algorithm variants produce valid results."""

    @pytest.mark.parametrize("algo", ["greedy", "ils", "reduced", "reduced+ils"])
    def test_complete_k4(self, algo):
        g = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(i, j)
        r = bmatching(g, capacities=1, algorithm=algo, time_limit=1.0)
        assert r.size == 2
        validate_bmatching(g, r, capacities=1)

    @pytest.mark.parametrize("algo", ["greedy", "ils", "reduced", "reduced+ils"])
    def test_path(self, algo):
        g = Graph(6)
        for i in range(5):
            g.add_edge(i, i + 1)
        r = bmatching(g, capacities=1, algorithm=algo, time_limit=1.0)
        assert r.size >= 2
        validate_bmatching(g, r, capacities=1)

    def test_invalid_algorithm(self):
        g = Graph(2)
        g.add_edge(0, 1)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            bmatching(g, algorithm="nonexistent")


# ---------------------------------------------------------------------------
# Hypergraph tests
# ---------------------------------------------------------------------------

class TestHypergraphBMatching:
    """Tests for hypergraph b-matching."""

    def test_simple_3uniform(self):
        """3-uniform hypergraph: 4 nodes, 2 edges."""
        edges = [[0, 1, 2], [1, 2, 3]]
        r = hypergraph_bmatching(4, edges, capacities=1, algorithm="greedy")
        # b=1: nodes 1,2 shared, so at most 1 edge
        assert r.size == 1
        validate_hypergraph_bmatching(4, edges, r, capacities=1)

    def test_disjoint_hyperedges(self):
        """Two disjoint 3-pin edges: both can be matched."""
        edges = [[0, 1, 2], [3, 4, 5]]
        r = hypergraph_bmatching(6, edges, capacities=1, algorithm="greedy")
        assert r.size == 2
        validate_hypergraph_bmatching(6, edges, r, capacities=1)

    def test_mixed_edge_sizes(self):
        """Mix of 2-pin and 3-pin edges."""
        edges = [[0, 1], [1, 2, 3], [4, 5]]
        r = hypergraph_bmatching(6, edges, capacities=1, algorithm="ils",
                                 time_limit=1.0)
        validate_hypergraph_bmatching(6, edges, r, capacities=1)

    def test_weighted_hyperedges(self):
        """Weighted hyperedges: prefer heavier edge."""
        edges = [[0, 1], [0, 2]]
        ew = np.array([10, 1], dtype=np.int32)
        r = hypergraph_bmatching(3, edges, capacities=1,
                                 edge_weights=ew, algorithm="greedy")
        assert r.weight >= 10
        validate_hypergraph_bmatching(3, edges, r, capacities=1)

    def test_capacity_constraints(self):
        """Higher capacity allows more incident edges."""
        # Star-like: node 0 in all edges
        edges = [[0, 1], [0, 2], [0, 3]]
        cap = np.array([2, 1, 1, 1], dtype=np.int32)
        r = hypergraph_bmatching(4, edges, capacities=cap,
                                 algorithm="ils", time_limit=1.0)
        assert r.size == 2
        validate_hypergraph_bmatching(4, edges, r, capacities=cap)

    @pytest.mark.parametrize("algo", ["greedy", "ils", "reduced", "reduced+ils"])
    def test_all_algorithms(self, algo):
        edges = [[0, 1, 2], [3, 4, 5], [0, 3]]
        r = hypergraph_bmatching(6, edges, capacities=1, algorithm=algo,
                                 time_limit=1.0)
        validate_hypergraph_bmatching(6, edges, r, capacities=1)
