"""Tests for KaMIS MIS algorithms."""

import numpy as np
import pytest

from chszlablib import Graph, IndependenceProblems


# ---------------------------------------------------------------------------
# Helper: build small test graphs
# ---------------------------------------------------------------------------

def make_path_graph(n):
    """Path graph: 0-1-2-..-(n-1). MIS size = ceil(n/2)."""
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def make_complete_graph(n):
    """Complete graph K_n. MIS size = 1."""
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def make_cycle_graph(n):
    """Cycle graph C_n. MIS size = floor(n/2)."""
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def make_weighted_path(n, weights):
    """Path graph with custom node weights."""
    g = Graph(n)
    for i in range(n):
        g.set_node_weight(i, weights[i])
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def is_valid_independent_set(g, vertices):
    """Check that no two vertices in the set are adjacent."""
    g.finalize()
    is_set = set(vertices)
    for u in is_set:
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            if g.adjncy[idx] in is_set:
                return False
    return True


# ---------------------------------------------------------------------------
# ReduMIS tests
# ---------------------------------------------------------------------------

class TestReduMIS:
    def test_path_graph(self):
        g = make_path_graph(10)
        r = IndependenceProblems.redumis(g, time_limit=5.0, seed=0)
        assert r.size >= 5  # optimal = 5
        assert is_valid_independent_set(g, r.vertices)

    def test_complete_graph(self):
        g = make_complete_graph(6)
        r = IndependenceProblems.redumis(g, time_limit=5.0, seed=0)
        assert r.size == 1
        assert is_valid_independent_set(g, r.vertices)

    def test_cycle_graph(self):
        g = make_cycle_graph(8)
        r = IndependenceProblems.redumis(g, time_limit=5.0, seed=0)
        assert r.size >= 4  # optimal = 4
        assert is_valid_independent_set(g, r.vertices)


# ---------------------------------------------------------------------------
# OnlineMIS tests
# ---------------------------------------------------------------------------

class TestOnlineMIS:
    def test_path_graph(self):
        g = make_path_graph(10)
        r = IndependenceProblems.online_mis(g, time_limit=5.0, seed=0, ils_iterations=5000)
        assert r.size >= 4  # should find near-optimal
        assert is_valid_independent_set(g, r.vertices)

    def test_complete_graph(self):
        g = make_complete_graph(6)
        r = IndependenceProblems.online_mis(g, time_limit=5.0, seed=0, ils_iterations=5000)
        assert r.size == 1
        assert is_valid_independent_set(g, r.vertices)


# ---------------------------------------------------------------------------
# Branch & Reduce tests (weighted)
# ---------------------------------------------------------------------------

class TestBranchReduce:
    def test_path_graph_weighted(self):
        # Path: 0-1-2-3-4 with weights [10, 1, 10, 1, 10]
        # Optimal MWIS = {0, 2, 4} with weight 30
        g = make_weighted_path(5, [10, 1, 10, 1, 10])
        r = IndependenceProblems.branch_reduce(g, time_limit=10.0, seed=0)
        assert r.weight == 30
        assert r.size == 3
        assert is_valid_independent_set(g, r.vertices)

    def test_complete_graph_weighted(self):
        g = make_complete_graph(5)
        for i in range(5):
            g.set_node_weight(i, i + 1)
        r = IndependenceProblems.branch_reduce(g, time_limit=10.0, seed=0)
        # Optimal: single heaviest vertex (weight 5)
        assert r.weight == 5
        assert r.size == 1
        assert is_valid_independent_set(g, r.vertices)

    def test_unweighted(self):
        g = make_path_graph(8)
        r = IndependenceProblems.branch_reduce(g, time_limit=10.0, seed=0)
        assert r.size >= 4  # optimal = 4
        assert is_valid_independent_set(g, r.vertices)


# ---------------------------------------------------------------------------
# MMWIS tests (weighted)
# ---------------------------------------------------------------------------

class TestMMWIS:
    def test_path_graph_weighted(self):
        g = make_weighted_path(5, [10, 1, 10, 1, 10])
        r = IndependenceProblems.mmwis(g, time_limit=5.0, seed=0)
        assert r.weight >= 20  # should find good solution, optimal = 30
        assert is_valid_independent_set(g, r.vertices)

    def test_complete_graph_weighted(self):
        g = make_complete_graph(5)
        for i in range(5):
            g.set_node_weight(i, i + 1)
        r = IndependenceProblems.mmwis(g, time_limit=5.0, seed=0)
        assert r.size == 1
        assert is_valid_independent_set(g, r.vertices)


# ---------------------------------------------------------------------------
# Unsorted-edge tests (from_csr with deliberately unsorted adjacency)
# ---------------------------------------------------------------------------

def _make_unsorted_triangle():
    """Triangle 0-1, 0-2, 1-2 with deliberately unsorted adjacency lists.

    Node 0's neighbors listed as [2, 1] instead of [1, 2].
    """
    xadj = np.array([0, 2, 4, 6], dtype=np.int64)
    adjncy = np.array([2, 1, 2, 0, 0, 1], dtype=np.int32)  # unsorted
    return Graph.from_csr(xadj, adjncy)


class TestUnsortedEdges:
    """Verify KaMIS algorithms work with unsorted adjacency lists (from_csr)."""

    def test_redumis_unsorted_csr(self):
        g = _make_unsorted_triangle()
        r = IndependenceProblems.redumis(g, time_limit=5.0, seed=0)
        assert r.size >= 1
        assert is_valid_independent_set(g, r.vertices)

    def test_online_mis_unsorted_csr(self):
        g = _make_unsorted_triangle()
        r = IndependenceProblems.online_mis(g, time_limit=5.0, seed=0,
                                            ils_iterations=1000)
        assert r.size >= 1
        assert is_valid_independent_set(g, r.vertices)

    def test_branch_reduce_unsorted_csr(self):
        g = _make_unsorted_triangle()
        r = IndependenceProblems.branch_reduce(g, time_limit=5.0, seed=0)
        assert r.size >= 1
        assert is_valid_independent_set(g, r.vertices)

    def test_mmwis_unsorted_csr(self):
        g = _make_unsorted_triangle()
        r = IndependenceProblems.mmwis(g, time_limit=5.0, seed=0)
        assert r.size >= 1
        assert is_valid_independent_set(g, r.vertices)
