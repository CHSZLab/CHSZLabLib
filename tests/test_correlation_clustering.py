"""Tests for ScalableCorrelationClustering."""

import numpy as np
import pytest

from chszlablib import Graph, Decomposition


# ---------------------------------------------------------------------------
# Helper: build signed test graphs
# ---------------------------------------------------------------------------

def make_two_cliques_signed(n1, n2):
    """Two cliques connected by negative edges.

    Nodes 0..n1-1 form one clique (positive edges),
    nodes n1..n1+n2-1 form another (positive edges),
    inter-clique edges are negative.
    """
    n = n1 + n2
    g = Graph(n)
    # Positive edges within first clique
    for i in range(n1):
        for j in range(i + 1, n1):
            g.add_edge(i, j, weight=1)
    # Positive edges within second clique
    for i in range(n1, n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=1)
    # Negative edges between cliques
    for i in range(n1):
        for j in range(n1, n):
            g.add_edge(i, j, weight=-1)
    return g


def make_all_positive(n):
    """Complete graph with all positive edges — should be one cluster."""
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=1)
    return g


def make_all_negative(n):
    """Complete graph with all negative edges — each node its own cluster."""
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=-1)
    return g


def make_signed_path(n):
    """Path graph with alternating +/- edges: +, -, +, -, ..."""
    g = Graph(n)
    for i in range(n - 1):
        w = 1 if i % 2 == 0 else -1
        g.add_edge(i, i + 1, weight=w)
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorrelationClustering:
    def test_two_cliques(self):
        """Two cliques with negative inter-edges should split into 2 clusters."""
        g = make_two_cliques_signed(4, 4)
        r = Decomposition.correlation_clustering(g, seed=0)
        # Should find at least 2 clusters
        assert r.num_clusters >= 2
        assert len(r.assignment) == 8
        # Nodes in same clique should be in same cluster
        clique1 = set(r.assignment[:4])
        clique2 = set(r.assignment[4:])
        # Each clique should have uniform assignment
        assert len(clique1) == 1
        assert len(clique2) == 1
        # The two cliques should be in different clusters
        assert clique1 != clique2

    def test_all_positive(self):
        """All positive edges — should put everything in one cluster."""
        g = make_all_positive(6)
        r = Decomposition.correlation_clustering(g, seed=0)
        assert r.num_clusters == 1
        assert r.edge_cut == 0

    def test_all_negative(self):
        """All negative edges — ideal: each node alone, edge_cut = 0."""
        g = make_all_negative(5)
        r = Decomposition.correlation_clustering(g, seed=0)
        # With all negative edges, optimal is singletons (edge_cut=0)
        # or at least the cut should be non-positive
        assert r.edge_cut <= 0

    def test_signed_path(self):
        """Alternating +/- path — should cluster along positive edges."""
        g = make_signed_path(6)
        r = Decomposition.correlation_clustering(g, seed=0)
        assert len(r.assignment) == 6

    def test_with_time_limit(self):
        """Run with time limit to exercise the multi-run path."""
        g = make_two_cliques_signed(3, 3)
        r = Decomposition.correlation_clustering(g, seed=0, time_limit=1.0)
        assert r.num_clusters >= 2
        assert len(r.assignment) == 6

    def test_assignment_valid(self):
        """Assignment values should be in [0, num_clusters)."""
        g = make_two_cliques_signed(4, 4)
        r = Decomposition.correlation_clustering(g, seed=0)
        assert all(0 <= a < r.num_clusters for a in r.assignment)


class TestEvolutionaryCorrelationClustering:
    def test_two_cliques(self):
        """Two cliques with negative inter-edges should split into 2 clusters."""
        g = make_two_cliques_signed(4, 4)
        r = Decomposition.evolutionary_correlation_clustering(g, seed=0, time_limit=2.0)
        assert r.num_clusters >= 2
        assert len(r.assignment) == 8
        clique1 = set(r.assignment[:4])
        clique2 = set(r.assignment[4:])
        assert len(clique1) == 1
        assert len(clique2) == 1
        assert clique1 != clique2

    def test_all_positive(self):
        """All positive edges — should put everything in one cluster."""
        g = make_all_positive(6)
        r = Decomposition.evolutionary_correlation_clustering(g, seed=0, time_limit=2.0)
        assert r.num_clusters == 1
        assert r.edge_cut == 0

    def test_all_negative(self):
        """All negative edges — ideal: each node alone, edge_cut = 0."""
        g = make_all_negative(5)
        r = Decomposition.evolutionary_correlation_clustering(g, seed=0, time_limit=2.0)
        assert r.edge_cut <= 0

    def test_assignment_valid(self):
        """Assignment values should be in [0, num_clusters)."""
        g = make_two_cliques_signed(4, 4)
        r = Decomposition.evolutionary_correlation_clustering(g, seed=0, time_limit=2.0)
        assert all(0 <= a < r.num_clusters for a in r.assignment)
