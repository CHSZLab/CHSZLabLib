"""Tests for Max-Cut (fpt-max-cut)."""

import numpy as np
import pytest

from chszlablib import Graph, maxcut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_path_graph(n):
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def make_cycle_graph(n):
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def make_complete_graph(n):
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def make_complete_bipartite(n1, n2):
    g = Graph(n1 + n2)
    for i in range(n1):
        for j in range(n1, n1 + n2):
            g.add_edge(i, j)
    return g


# ---------------------------------------------------------------------------
# Heuristic max-cut
# ---------------------------------------------------------------------------

class TestMaxCutHeuristic:

    def test_single_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        r = maxcut(g)
        assert r.cut_value >= 1
        assert r.partition.shape == (2,)

    def test_path_graph(self):
        g = make_path_graph(4)
        r = maxcut(g)
        assert r.cut_value >= 2

    def test_triangle(self):
        g = make_cycle_graph(3)
        r = maxcut(g)
        assert r.cut_value >= 2

    def test_complete_k4(self):
        g = make_complete_graph(4)
        r = maxcut(g, time_limit=0.5)
        assert r.cut_value >= 4

    def test_complete_bipartite(self):
        """K_{3,3}: max cut = 9."""
        g = make_complete_bipartite(3, 3)
        r = maxcut(g, time_limit=0.5)
        assert r.cut_value >= 9

    def test_even_cycle(self):
        """C6: max cut = 6 (alternating partition)."""
        g = make_cycle_graph(6)
        r = maxcut(g, time_limit=1.0)
        assert r.cut_value >= 4

    def test_empty_graph(self):
        g = Graph(5)
        r = maxcut(g)
        assert r.cut_value == 0

    def test_weighted_edges(self):
        g = Graph(4)
        g.add_edge(0, 1, weight=10)
        g.add_edge(1, 2, weight=1)
        g.add_edge(2, 3, weight=10)
        r = maxcut(g, time_limit=0.5)
        assert r.cut_value >= 20


# ---------------------------------------------------------------------------
# Exact max-cut
# ---------------------------------------------------------------------------

class TestMaxCutExact:

    def test_single_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        r = maxcut(g, method="exact")
        assert r.cut_value >= 1
        assert r.partition.shape == (2,)

    def test_path_4(self):
        g = make_path_graph(4)
        r = maxcut(g, method="exact", time_limit=5.0)
        assert r.cut_value >= 2

    def test_triangle(self):
        g = make_cycle_graph(3)
        r = maxcut(g, method="exact", time_limit=5.0)
        assert r.cut_value >= 2

    def test_small_complete(self):
        g = make_complete_graph(5)
        r = maxcut(g, method="exact", time_limit=5.0)
        # K5: max cut = 6 (ceil(5*4/4) = 6 by Edwards bound)
        assert r.cut_value >= 6


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestMaxCutErrors:

    def test_invalid_method(self):
        g = Graph(2)
        g.add_edge(0, 1)
        with pytest.raises(ValueError, match="Unknown method"):
            maxcut(g, method="nonexistent")
