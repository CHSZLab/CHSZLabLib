"""Integration tests using all four algorithm families on the same graphs."""

import os

import numpy as np
import pytest

from chszlablib import Graph, Decomposition, IndependenceProblems

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def kahip_example():
    path = os.path.join(REPO_ROOT, "external_repositories", "KaHIP", "examples", "rgg_n_2_15_s0.graph")
    if not os.path.exists(path):
        pytest.skip("KaHIP example graph not found")
    return Graph.from_metis(path)


def make_two_cliques_bridge():
    """Two K4 subgraphs connected by a single bridge edge (weight 1).
    Internal edges have weight 5, node weights are 10."""
    g = Graph(num_nodes=8)
    for i in range(8):
        g.set_node_weight(i, 10)
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j, weight=5)
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j, weight=5)
    g.add_edge(3, 4, weight=1)
    return g


def test_all_algorithms_on_same_graph():
    """Run all four algorithm families on the same graph."""
    g = make_two_cliques_bridge()

    # Partition into 2 parts
    pr = Decomposition.partition(g, num_parts=2, mode="fast")
    assert pr.edgecut >= 0
    assert len(pr.assignment) == 8

    # Minimum cut (should be 1 at the bridge)
    mc = Decomposition.mincut(g, algorithm="noi")
    assert mc.cut_value == 1
    assert len(mc.partition) == 8

    # Clustering (should find 2 communities)
    cr = Decomposition.cluster(g, time_limit=1.0)
    assert cr.modularity > 0
    assert len(cr.assignment) == 8

    # MWIS
    mr = IndependenceProblems.chils(g, time_limit=1.0, num_concurrent=1)
    assert mr.weight > 0
    # Verify independent set validity
    is_set = set(mr.vertices)
    g.finalize()
    for u in is_set:
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            assert g.adjncy[idx] not in is_set, (
                f"Vertices {u} and {g.adjncy[idx]} are adjacent but both in IS"
            )


def test_metis_roundtrip_with_algorithm(tmp_path):
    """Build graph, save to METIS, reload, run algorithm."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)

    path = str(tmp_path / "k4.graph")
    g.to_metis(path)
    g2 = Graph.from_metis(path)

    result = Decomposition.partition(g2, num_parts=2, mode="fast")
    assert result.edgecut >= 0
    assert len(result.assignment) == 4


def test_large_kahip_example(kahip_example):
    """Partition the KaHIP example graph (32K nodes)."""
    result = Decomposition.partition(kahip_example, num_parts=4, mode="fast")
    assert result.edgecut > 0
    assert len(result.assignment) == kahip_example.num_nodes
