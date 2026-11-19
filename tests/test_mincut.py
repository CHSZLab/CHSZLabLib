"""Tests for the global minimum cut wrapper (chszlablib.mincut)."""

import numpy as np
import pytest

from chszlablib import Graph, Decomposition


def make_barbell():
    """Two K3 connected by a single edge.  Min cut = 1."""
    g = Graph(num_nodes=6)
    # First triangle
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    # Bridge
    g.add_edge(2, 3)
    # Second triangle
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)
    return g


def test_mincut_barbell():
    g = make_barbell()
    result = Decomposition.mincut(g, algorithm="inexact")
    assert result.cut_value == 1
    assert len(result.partition) == 6
    assert set(np.unique(result.partition)) == {0, 1}


def test_mincut_complete_graph():
    """K4 has minimum cut = 3."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    result = Decomposition.mincut(g, algorithm="inexact")
    assert result.cut_value == 3


def test_mincut_weighted():
    """Path 0--1--2 with weights 10 and 1.  Min cut = 1."""
    g = Graph(num_nodes=3)
    g.add_edge(0, 1, weight=10)
    g.add_edge(1, 2, weight=1)
    result = Decomposition.mincut(g, algorithm="inexact")
    assert result.cut_value == 1


@pytest.mark.parametrize("algo", ["inexact", "exact", "cactus", "sw"])
def test_mincut_algorithms(algo):
    """Verify multiple algorithms agree on the barbell graph."""
    g = make_barbell()
    result = Decomposition.mincut(g, algorithm=algo)
    assert result.cut_value == 1, f"Algorithm {algo} returned {result.cut_value}"


def test_mincut_sw_matches_exact():
    """Stoer-Wagner agrees with the exact algorithm on a random weighted graph."""
    rng = np.random.default_rng(7)
    n = 60
    g = Graph(num_nodes=n)
    for i in range(n - 1):  # path keeps the graph connected
        g.add_edge(i, i + 1, weight=int(rng.integers(1, 10)))
    extra = set()
    while len(extra) < 120:
        u, v = sorted(int(x) for x in rng.integers(0, n, size=2))
        if v - u > 1:
            extra.add((u, v))
    for u, v in sorted(extra):
        g.add_edge(u, v, weight=int(rng.integers(1, 10)))
    sw = Decomposition.mincut(g, algorithm="sw")
    exact = Decomposition.mincut(g, algorithm="exact")
    assert sw.cut_value == exact.cut_value
    # The returned side must realize the reported cut value.
    side = sw.partition
    assert 0 < side.sum() < n
    g.finalize()
    cut = sum(
        int(w) for u in range(n)
        for v, w in zip(g.adjncy[g.xadj[u]:g.xadj[u + 1]], g.edge_weights[g.xadj[u]:g.xadj[u + 1]])
        if side[u] != side[v]
    ) // 2
    assert cut == sw.cut_value
