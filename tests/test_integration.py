"""Integration tests using all four algorithm families on the same graphs."""

import os

import numpy as np
import pytest

from chszlablib import Graph, HyperGraph, Decomposition, IndependenceProblems

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
    mc = Decomposition.mincut(g, algorithm="inexact")
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


def test_hypergraph_to_graph_then_mis():
    """Build hypergraph, run HyperMIS, expand to graph, run graph MIS."""
    hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4], [4, 5]])

    # HyperMIS on the hypergraph
    hr = IndependenceProblems.hypermis(hg, time_limit=5.0)
    assert hr.size >= 1
    hyper_selected = set(hr.vertices.tolist())
    for eid in range(hg.num_edges):
        s, e = hg.eptr[eid], hg.eptr[eid + 1]
        assert len(hyper_selected & set(hg.everts[s:e].tolist())) <= 1

    # Expand to graph and run graph MIS
    g = hg.to_graph()
    gr = IndependenceProblems.redumis(g, time_limit=2.0)
    assert gr.size >= 1
    graph_selected = set(gr.vertices.tolist())
    g.finalize()
    for u in graph_selected:
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            assert g.adjncy[idx] not in graph_selected


def test_hmetis_roundtrip_with_hypermis(tmp_path):
    """Save hypergraph to hMETIS, reload, run HyperMIS."""
    hg = HyperGraph.from_edge_list([[0, 1], [1, 2], [2, 3, 4]])
    path = str(tmp_path / "test.hgr")
    hg.to_hmetis(path)
    hg2 = HyperGraph.from_hmetis(path)
    result = IndependenceProblems.hypermis(hg2, time_limit=5.0)
    assert result.size >= 1
    assert len(result.vertices) == result.size


def test_large_kahip_example(kahip_example):
    """Partition the KaHIP example graph (32K nodes)."""
    result = Decomposition.partition(kahip_example, num_parts=4, mode="fast")
    assert result.edgecut > 0
    assert len(result.assignment) == kahip_example.num_nodes
