"""Tests for the CHILS maximum weight independent set wrapper."""

import numpy as np
import pytest

from chszlablib import Graph, IndependenceProblems, MWISResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_path_weighted():
    """Weighted path: 0(10)--1(1)--2(10). Optimal IS = {0, 2}, weight = 20."""
    g = Graph(num_nodes=3)
    g.set_node_weight(0, 10)
    g.set_node_weight(1, 1)
    g.set_node_weight(2, 10)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_mwis_path():
    g = make_path_weighted()
    result = IndependenceProblems.chils(g, time_limit=1.0, num_concurrent=1)
    assert isinstance(result, MWISResult)
    assert result.weight == 20
    assert set(result.vertices) == {0, 2}


def test_mwis_independent_set_valid():
    g = Graph(num_nodes=5)
    for i in range(5):
        g.set_node_weight(i, i + 1)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.finalize()

    result = IndependenceProblems.chils(g, time_limit=1.0, num_concurrent=1)

    # Verify independent set property: no two selected vertices are adjacent
    is_set = set(result.vertices)
    for u in is_set:
        start = g.xadj[u]
        end = g.xadj[u + 1]
        for idx in range(start, end):
            assert g.adjncy[idx] not in is_set


def test_mwis_single_node():
    g = Graph(num_nodes=1)
    g.set_node_weight(0, 42)
    result = IndependenceProblems.chils(g, time_limit=1.0, num_concurrent=1)
    assert result.weight == 42
    assert list(result.vertices) == [0]
