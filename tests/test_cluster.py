"""Tests for VieClus graph clustering."""

import numpy as np
import pytest

from chszlablib import Graph, Decomposition


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
    result = Decomposition.cluster(g, time_limit=1.0)
    assert result.modularity > 0.0
    assert result.num_clusters >= 2
    assert len(result.assignment) == 8


def test_cluster_modularity_positive():
    g = make_two_cliques()
    result = Decomposition.cluster(g, time_limit=1.0)
    assert result.modularity > 0.3


def _clusters_connected(g, assignment):
    """True if every cluster induces a connected subgraph."""
    xadj, adjncy = g.xadj, g.adjncy
    for c in np.unique(assignment):
        members = np.flatnonzero(assignment == c)
        seen = {int(members[0])}
        stack = [int(members[0])]
        while stack:
            v = stack.pop()
            for u in adjncy[xadj[v]:xadj[v + 1]]:
                u = int(u)
                if assignment[u] == c and u not in seen:
                    seen.add(u)
                    stack.append(u)
        if len(seen) != len(members):
            return False
    return True


def _grid_graph(rows, cols):
    g = Graph(num_nodes=rows * cols)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                g.add_edge(v, v + 1)
            if r + 1 < rows:
                g.add_edge(v, v + cols)
    return g


def test_cluster_leiden_two_cliques():
    g = make_two_cliques()
    result = Decomposition.cluster(g, time_limit=1.0, leiden=True)
    assert result.modularity > 0.3
    assert result.num_clusters >= 2
    assert _clusters_connected(g, result.assignment)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cluster_leiden_connected_on_grid(seed):
    g = _grid_graph(20, 20)
    result = Decomposition.cluster(g, time_limit=1.0, seed=seed, leiden=True)
    assert len(result.assignment) == 400
    assert result.num_clusters == len(np.unique(result.assignment))
    assert result.modularity > 0.5
    assert _clusters_connected(g, result.assignment)


def test_cluster_leiden_theta():
    g = _grid_graph(10, 10)
    result = Decomposition.cluster(g, time_limit=0.5, leiden=True, leiden_theta=0.1)
    assert _clusters_connected(g, result.assignment)


def test_cluster_leiden_invalid_theta():
    g = make_two_cliques()
    with pytest.raises(ValueError):
        Decomposition.cluster(g, leiden=True, leiden_theta=0.0)
