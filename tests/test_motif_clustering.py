"""Tests for HeidelbergMotifClustering (SOCIAL and LMCHGP)."""

import numpy as np
import pytest

from chszlablib import Graph, motif_cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_triangle():
    """Triangle: 0-1-2-0."""
    g = Graph(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)
    return g


def make_two_triangles_shared_edge():
    """Two triangles sharing edge 1-2: 0-1-2-0 and 1-2-3-1."""
    g = Graph(4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    return g


def make_barbell():
    """Two K4 cliques connected by a bridge edge."""
    g = Graph(8)
    # First K4: 0,1,2,3
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    # Second K4: 4,5,6,7
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j)
    # Bridge
    g.add_edge(3, 4)
    return g


def make_star(n):
    """Star graph with center 0 and leaves 1..n-1."""
    g = Graph(n)
    for i in range(1, n):
        g.add_edge(0, i)
    return g


# ---------------------------------------------------------------------------
# SOCIAL method
# ---------------------------------------------------------------------------

class TestMotifClusterSocial:

    def test_triangle(self):
        g = make_triangle()
        r = motif_cluster(g, seed_node=0, method="social",
                          bfs_depths=[5], time_limit=10)
        assert r.cluster_nodes is not None
        assert r.motif_conductance >= 0

    def test_two_triangles(self):
        g = make_two_triangles_shared_edge()
        r = motif_cluster(g, seed_node=0, method="social",
                          bfs_depths=[5], time_limit=10)
        assert len(r.cluster_nodes) > 0
        assert r.motif_conductance >= 0

    def test_barbell_finds_clique(self):
        """Seed in first K4 should find cluster in that clique."""
        g = make_barbell()
        r = motif_cluster(g, seed_node=0, method="social",
                          bfs_depths=[5, 10], time_limit=10)
        assert len(r.cluster_nodes) > 0
        # Seed node should be in the cluster
        assert 0 in r.cluster_nodes

    def test_default_bfs_depths(self):
        g = make_two_triangles_shared_edge()
        r = motif_cluster(g, seed_node=0)
        assert r.cluster_nodes is not None


# ---------------------------------------------------------------------------
# LMCHGP method
# ---------------------------------------------------------------------------

class TestMotifClusterLMCHGP:

    def test_triangle(self):
        g = make_triangle()
        r = motif_cluster(g, seed_node=0, method="lmchgp",
                          bfs_depths=[5], time_limit=10)
        assert r.cluster_nodes is not None
        assert r.motif_conductance >= 0

    def test_two_triangles(self):
        g = make_two_triangles_shared_edge()
        r = motif_cluster(g, seed_node=0, method="lmchgp",
                          bfs_depths=[5], time_limit=10)
        assert len(r.cluster_nodes) > 0
        assert r.motif_conductance >= 0

    def test_barbell_finds_clique(self):
        g = make_barbell()
        r = motif_cluster(g, seed_node=0, method="lmchgp",
                          bfs_depths=[5, 10], time_limit=10)
        assert len(r.cluster_nodes) > 0
        assert 0 in r.cluster_nodes


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestMotifClusterErrors:

    def test_invalid_method(self):
        g = make_triangle()
        with pytest.raises(ValueError, match="Unknown method"):
            motif_cluster(g, seed_node=0, method="nonexistent")
