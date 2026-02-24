"""Tests for the KaHIP evolutionary partitioner (kaffpaE) wrapper."""

import numpy as np
import pytest

from chszlablib.graph import Graph
from chszlablib.partition import PartitionResult, kaffpaE


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_bipartite():
    """Complete bipartite graph K_{3,3}: nodes 0-2 on one side, 3-5 on the other."""
    g = Graph(num_nodes=6)
    for i in range(3):
        for j in range(3, 6):
            g.add_edge(i, j)
    return g


def make_path():
    """Path graph: 0--1--2--3--4."""
    g = Graph(num_nodes=5)
    for i in range(4):
        g.add_edge(i, i + 1)
    return g


# ------------------------------------------------------------------
# kaffpaE tests
# ------------------------------------------------------------------

class TestKaffpaE:
    def test_basic(self):
        g = make_bipartite()
        result = kaffpaE(g, num_parts=2, time_limit=1, mode="fast")
        assert isinstance(result, PartitionResult)
        assert result.edgecut >= 0
        assert result.balance is not None
        assert result.balance >= 1.0
        assert len(result.assignment) == 6
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_modes(self):
        g = make_bipartite()
        for mode in ["fast", "eco", "strong"]:
            result = kaffpaE(g, num_parts=2, time_limit=1, mode=mode)
            assert result.edgecut >= 0
            assert len(result.assignment) == 6

    def test_social_modes(self):
        g = make_bipartite()
        for mode in ["fastsocial", "ecosocial", "strongsocial", "ultrafastsocial"]:
            result = kaffpaE(g, num_parts=2, time_limit=1, mode=mode)
            assert result.edgecut >= 0

    def test_k_parts(self):
        g = make_bipartite()
        result = kaffpaE(g, num_parts=3, time_limit=1, mode="fast")
        assert len(result.assignment) == 6
        assert len(np.unique(result.assignment)) <= 3

    def test_path(self):
        g = make_path()
        result = kaffpaE(g, num_parts=2, time_limit=1, mode="fast")
        assert result.edgecut >= 1
        assert len(result.assignment) == 5

    def test_warm_start(self):
        g = make_bipartite()
        # First run without warm start
        r1 = kaffpaE(g, num_parts=2, time_limit=1, mode="fast", seed=42)
        # Warm-start from the first result
        r2 = kaffpaE(
            g, num_parts=2, time_limit=1, mode="fast", seed=42,
            initial_partition=r1.assignment,
        )
        assert r2.edgecut >= 0
        assert len(r2.assignment) == 6

    def test_invalid_mode(self):
        g = make_bipartite()
        with pytest.raises(KeyError):
            kaffpaE(g, num_parts=2, time_limit=1, mode="nonexistent")

    def test_balance_field(self):
        """PartitionResult from kaffpaE includes balance."""
        g = make_bipartite()
        result = kaffpaE(g, num_parts=2, time_limit=1, mode="fast")
        assert isinstance(result.balance, float)
