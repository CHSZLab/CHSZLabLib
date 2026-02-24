"""Tests for KaLP longest path algorithm."""

import numpy as np
import pytest

from chszlablib import Graph, PathProblems


# ---------------------------------------------------------------------------
# Helper: build small test graphs
# ---------------------------------------------------------------------------

def make_path_graph(n):
    """Path graph: 0-1-2-..-(n-1). Longest path = n-1 edges."""
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def make_cycle_graph(n):
    """Cycle graph C_n. Longest path from 0 to any other = n-1 edges."""
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def make_grid_graph(rows, cols):
    """Grid graph with rows*cols vertices."""
    g = Graph(rows * cols)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                g.add_edge(v, v + 1)
            if r + 1 < rows:
                g.add_edge(v, v + cols)
    return g


def make_weighted_path(n, weights):
    """Path graph with edge weights."""
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1, weights[i])
    return g


def is_valid_path(g, path):
    """Check that path is a valid simple path in the graph."""
    if len(path) == 0:
        return True
    g.finalize()
    # All vertices unique
    if len(set(path)) != len(path):
        return False
    # Check adjacency
    adj = set()
    for u in range(len(g.xadj) - 1):
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            adj.add((u, int(g.adjncy[idx])))
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) not in adj:
            return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLongestPath:
    def test_path_graph(self):
        g = make_path_graph(8)
        r = PathProblems.longest_path(g, start_vertex=0, target_vertex=7)
        assert r.length == 7  # exactly n-1 edges
        assert len(r.path) == 8
        assert r.path[0] == 0
        assert r.path[-1] == 7
        assert is_valid_path(g, r.path)

    def test_path_graph_default_target(self):
        g = make_path_graph(6)
        r = PathProblems.longest_path(g, start_vertex=0)
        assert r.length == 5
        assert is_valid_path(g, r.path)

    def test_cycle_graph(self):
        g = make_cycle_graph(6)
        r = PathProblems.longest_path(g, start_vertex=0, target_vertex=3)
        # Longest path in C6 from 0 to 3 = 5 edges (go around)
        assert r.length >= 3  # at minimum the short way
        assert is_valid_path(g, r.path)

    def test_grid_graph(self):
        g = make_grid_graph(3, 3)
        r = PathProblems.longest_path(g, start_vertex=0, target_vertex=8)
        assert r.length >= 4  # at least shortest path
        assert is_valid_path(g, r.path)

    def test_no_path(self):
        # Disconnected graph: two components
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        r = PathProblems.longest_path(g, start_vertex=0, target_vertex=3)
        assert r.length == 0
        assert len(r.path) == 0

    def test_weighted_path(self):
        g = make_weighted_path(5, [1, 10, 1, 10])
        r = PathProblems.longest_path(g, start_vertex=0, target_vertex=4)
        assert r.length == 22  # sum of all edge weights
        assert is_valid_path(g, r.path)


class TestLongestPathConfigs:
    @pytest.mark.parametrize("config", ["strong", "eco", "fast"])
    def test_partition_configs(self, config):
        g = make_path_graph(10)
        r = PathProblems.longest_path(g, start_vertex=0, target_vertex=9,
                         partition_config=config)
        assert r.length == 9
        assert is_valid_path(g, r.path)

    def test_invalid_config(self):
        g = make_path_graph(5)
        with pytest.raises(ValueError, match="Invalid partition_config"):
            PathProblems.longest_path(g, partition_config="invalid")

    def test_invalid_start_vertex(self):
        g = make_path_graph(5)
        with pytest.raises(ValueError, match="start_vertex"):
            PathProblems.longest_path(g, start_vertex=10)

    def test_invalid_target_vertex(self):
        g = make_path_graph(5)
        with pytest.raises(ValueError, match="target_vertex"):
            PathProblems.longest_path(g, target_vertex=10)
