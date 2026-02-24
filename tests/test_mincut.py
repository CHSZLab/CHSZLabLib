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
    result = Decomposition.mincut(g, algorithm="noi")
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
    result = Decomposition.mincut(g, algorithm="noi")
    assert result.cut_value == 3


def test_mincut_weighted():
    """Path 0--1--2 with weights 10 and 1.  Min cut = 1."""
    g = Graph(num_nodes=3)
    g.add_edge(0, 1, weight=10)
    g.add_edge(1, 2, weight=1)
    result = Decomposition.mincut(g, algorithm="noi")
    assert result.cut_value == 1


@pytest.mark.parametrize("algo", ["noi", "viecut", "pr"])
def test_mincut_algorithms(algo):
    """Verify multiple algorithms agree on the barbell graph."""
    g = make_barbell()
    result = Decomposition.mincut(g, algorithm=algo)
    assert result.cut_value == 1, f"Algorithm {algo} returned {result.cut_value}"
