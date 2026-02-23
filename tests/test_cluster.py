"""Tests for VieClus graph clustering."""

import numpy as np
import pytest

from chszlablib.graph import Graph
from chszlablib.cluster import cluster


def make_two_cliques():
    """Create a graph with two 4-cliques connected by a single bridge edge."""
    g = Graph(num_nodes=8)
    # First clique: nodes 0-3
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    # Second clique: nodes 4-7
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j)
    # Bridge edge
    g.add_edge(3, 4)
    return g


def test_cluster_two_cliques():
    g = make_two_cliques()
    result = cluster(g, time_limit=1.0)
    assert result.modularity > 0.0
    assert result.num_clusters >= 2
    assert len(result.assignment) == 8


def test_cluster_modularity_positive():
    g = make_two_cliques()
    result = cluster(g, time_limit=1.0)
    assert result.modularity > 0.3
