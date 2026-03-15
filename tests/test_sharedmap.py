"""Tests for SharedMap (hierarchical process mapping) via Decomposition.process_map."""

import numpy as np
import pytest

from chszlablib import Graph, Decomposition, ProcessMappingResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_square():
    """Square graph: 0--1--2--3--0 with uniform weights."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1, 10)
    g.add_edge(1, 2, 20)
    g.add_edge(2, 3, 10)
    g.add_edge(3, 0, 20)
    return g


def make_path():
    """Path graph: 0--1--2--3--4--5--6--7 with unit weights."""
    g = Graph(num_nodes=8)
    for i in range(7):
        g.add_edge(i, i + 1)
    return g


# ------------------------------------------------------------------
# process_map tests
# ------------------------------------------------------------------

class TestProcessMap:
    def test_basic(self):
        g = make_square()
        result = Decomposition.process_map(
            g, hierarchy=[2, 2], distance=[1, 10], mode="fast"
        )
        assert isinstance(result, ProcessMappingResult)
        assert result.comm_cost >= 0
        assert len(result.assignment) == 4
        # Each node mapped to one of 2*2=4 PEs
        assert all(0 <= a < 4 for a in result.assignment)

    def test_modes(self):
        g = make_square()
        for mode in ("fast", "eco", "strong"):
            result = Decomposition.process_map(
                g, hierarchy=[2, 2], distance=[1, 10], mode=mode
            )
            assert result.comm_cost >= 0
            assert len(result.assignment) == 4

    def test_path_graph(self):
        g = make_path()
        result = Decomposition.process_map(
            g, hierarchy=[2, 4], distance=[10, 1], mode="fast"
        )
        assert result.comm_cost >= 0
        assert len(result.assignment) == 8
        assert all(0 <= a < 8 for a in result.assignment)

    def test_seed_reproducibility(self):
        g = make_square()
        r1 = Decomposition.process_map(
            g, hierarchy=[2, 2], distance=[1, 10], mode="fast", seed=42
        )
        r2 = Decomposition.process_map(
            g, hierarchy=[2, 2], distance=[1, 10], mode="fast", seed=42
        )
        np.testing.assert_array_equal(r1.assignment, r2.assignment)
        assert r1.comm_cost == r2.comm_cost

    def test_invalid_mode(self):
        g = make_square()
        with pytest.raises(ValueError, match="Unknown mode"):
            Decomposition.process_map(
                g, hierarchy=[2, 2], distance=[1, 10], mode="nonexistent"
            )

    def test_mismatched_hierarchy_distance(self):
        g = make_square()
        with pytest.raises(ValueError, match="same length"):
            Decomposition.process_map(
                g, hierarchy=[2, 2], distance=[1], mode="fast"
            )
