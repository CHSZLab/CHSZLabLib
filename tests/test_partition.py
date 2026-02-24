"""Tests for the KaHIP partition, node_separator, and node_ordering wrappers."""

import numpy as np
import pytest

from chszlablib import Graph, Decomposition, PartitionResult, SeparatorResult, OrderingResult


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
# partition tests
# ------------------------------------------------------------------

class TestPartition:
    def test_partition_bipartite(self):
        g = make_bipartite()
        result = Decomposition.partition(g, num_parts=2, mode="strong")
        assert isinstance(result, PartitionResult)
        assert result.edgecut >= 0
        assert len(result.assignment) == 6
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_partition_modes(self):
        g = make_bipartite()
        for mode in ["fast", "eco", "strong"]:
            result = Decomposition.partition(g, num_parts=2, mode=mode)
            assert result.edgecut >= 0
            assert len(result.assignment) == 6

    def test_partition_social_modes(self):
        g = make_bipartite()
        for mode in ["fastsocial", "ecosocial", "strongsocial"]:
            result = Decomposition.partition(g, num_parts=2, mode=mode)
            assert result.edgecut >= 0

    def test_partition_k_parts(self):
        g = make_bipartite()
        result = Decomposition.partition(g, num_parts=3, mode="fast")
        assert len(result.assignment) == 6
        assert len(np.unique(result.assignment)) <= 3

    def test_partition_path(self):
        g = make_path()
        result = Decomposition.partition(g, num_parts=2, mode="eco")
        assert result.edgecut >= 1
        assert len(result.assignment) == 5

    def test_partition_seed_reproducibility(self):
        g = make_bipartite()
        r1 = Decomposition.partition(g, num_parts=2, mode="fast", seed=42)
        r2 = Decomposition.partition(g, num_parts=2, mode="fast", seed=42)
        np.testing.assert_array_equal(r1.assignment, r2.assignment)
        assert r1.edgecut == r2.edgecut

    def test_partition_invalid_mode(self):
        g = make_bipartite()
        with pytest.raises(KeyError):
            Decomposition.partition(g, num_parts=2, mode="nonexistent")


# ------------------------------------------------------------------
# node_separator tests
# ------------------------------------------------------------------

class TestNodeSeparator:
    def test_node_separator_basic(self):
        g = make_bipartite()
        result = Decomposition.node_separator(g, num_parts=2, mode="strong")
        assert isinstance(result, SeparatorResult)
        assert result.num_separator_vertices >= 0
        assert len(result.separator) == result.num_separator_vertices

    def test_node_separator_path(self):
        g = make_path()
        result = Decomposition.node_separator(g, num_parts=2, mode="eco")
        assert result.num_separator_vertices >= 0
        # Separator nodes must be valid node indices
        for v in result.separator:
            assert 0 <= v < 5


# ------------------------------------------------------------------
# node_ordering tests
# ------------------------------------------------------------------

class TestNodeOrdering:
    def test_node_ordering_basic(self):
        g = make_bipartite()
        result = Decomposition.node_ordering(g, mode="fast")
        assert isinstance(result, OrderingResult)
        assert len(result.ordering) == 6
        # Must be a permutation of 0..n-1
        assert set(result.ordering) == set(range(6))

    def test_node_ordering_path(self):
        g = make_path()
        result = Decomposition.node_ordering(g, mode="eco")
        assert len(result.ordering) == 5
        assert set(result.ordering) == set(range(5))
