"""Tests for edge orientation (HeiOrient)."""

import math

import numpy as np
import pytest

from chszlablib import Graph, orient_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_path_graph(n):
    """Path: 0-1-2-...-(n-1)."""
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def make_cycle_graph(n):
    """Cycle: 0-1-2-...-(n-1)-0."""
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def make_star_graph(n):
    """Star: center=0, leaves=1..n-1."""
    g = Graph(n)
    for i in range(1, n):
        g.add_edge(0, i)
    return g


def make_complete_graph(n):
    """Complete graph K_n."""
    g = Graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def validate_orientation(g, r):
    """Check that edge_heads and out_degrees are consistent."""
    g.finalize()
    n = g.num_nodes

    # out_degrees must match edge_heads sums
    for u in range(n):
        start, end = g.xadj[u], g.xadj[u + 1]
        assert r.out_degrees[u] == np.sum(r.edge_heads[start:end]), (
            f"Node {u}: out_degrees={r.out_degrees[u]} != "
            f"sum(edge_heads)={np.sum(r.edge_heads[start:end])}"
        )

    # max_out_degree must match
    assert r.max_out_degree == int(np.max(r.out_degrees)) if n > 0 else True

    # For each undirected edge (u, v), exactly one direction must be 1
    for u in range(n):
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            v = g.adjncy[idx]
            if u < v:
                # Find v->u entry
                head_uv = r.edge_heads[idx]
                found = False
                for idx2 in range(g.xadj[v], g.xadj[v + 1]):
                    if g.adjncy[idx2] == u:
                        head_vu = r.edge_heads[idx2]
                        found = True
                        break
                assert found, f"Edge ({u},{v}): reverse not found in CSR"
                assert head_uv + head_vu == 1, (
                    f"Edge ({u},{v}): head_uv={head_uv}, head_vu={head_vu} "
                    f"(should sum to 1)"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrientEdges:
    """Basic algorithm tests."""

    def test_single_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        r = orient_edges(g)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    def test_triangle(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        r = orient_edges(g)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    def test_path_graph(self):
        g = make_path_graph(10)
        r = orient_edges(g)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    def test_even_cycle(self):
        g = make_cycle_graph(6)
        r = orient_edges(g)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    def test_odd_cycle(self):
        g = make_cycle_graph(7)
        r = orient_edges(g)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    def test_star_graph(self):
        g = make_star_graph(6)
        r = orient_edges(g)
        # Star with 5 leaves: optimal max out-degree = 1
        # (orient all edges toward center)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    def test_complete_k4(self):
        g = make_complete_graph(4)
        r = orient_edges(g)
        # K4: 6 edges, 4 nodes -> ceil(6/4) = 2
        # But a tournament on K4 has max out-degree = ceil((4-1)/2) = 2
        assert r.max_out_degree <= 2
        validate_orientation(g, r)

    def test_complete_k5(self):
        g = make_complete_graph(5)
        r = orient_edges(g)
        # K5: 10 edges, 5 nodes -> ceil(10/5) = 2
        assert r.max_out_degree == 2
        validate_orientation(g, r)

    def test_result_shapes(self):
        g = make_path_graph(8)
        r = orient_edges(g)
        g.finalize()
        assert r.out_degrees.shape == (8,)
        assert r.edge_heads.shape == (len(g.adjncy),)


class TestAlgorithms:
    """Test all three algorithm variants."""

    @pytest.mark.parametrize("algo", ["two_approx", "dfs", "combined"])
    def test_path(self, algo):
        g = make_path_graph(20)
        r = orient_edges(g, algorithm=algo)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    @pytest.mark.parametrize("algo", ["two_approx", "dfs", "combined"])
    def test_cycle(self, algo):
        g = make_cycle_graph(10)
        r = orient_edges(g, algorithm=algo)
        assert r.max_out_degree == 1
        validate_orientation(g, r)

    @pytest.mark.parametrize("algo", ["two_approx", "dfs", "combined"])
    def test_complete(self, algo):
        g = make_complete_graph(6)
        r = orient_edges(g, algorithm=algo)
        # K6: 15 edges, 6 nodes -> ceil(15/6) = 3
        lower_bound = math.ceil(15 / 6)
        assert r.max_out_degree >= lower_bound
        validate_orientation(g, r)

    def test_invalid_algorithm(self):
        g = make_path_graph(4)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            orient_edges(g, algorithm="nonexistent")


class TestLarger:
    """Slightly larger graphs to stress-test correctness."""

    def test_grid_graph(self):
        """4x4 grid: 16 nodes, 24 edges -> lower bound ceil(24/16)=2."""
        n = 16
        g = Graph(n)
        for r in range(4):
            for c in range(4):
                u = r * 4 + c
                if c < 3:
                    g.add_edge(u, u + 1)
                if r < 3:
                    g.add_edge(u, u + 4)
        r = orient_edges(g)
        assert r.max_out_degree <= 2
        validate_orientation(g, r)

    def test_petersen_graph(self):
        """Petersen graph: 10 nodes, 15 edges, 3-regular -> lower bound 2."""
        g = Graph(10)
        # Outer cycle
        for i in range(5):
            g.add_edge(i, (i + 1) % 5)
        # Inner pentagram
        for i in range(5):
            g.add_edge(5 + i, 5 + (i + 2) % 5)
        # Spokes
        for i in range(5):
            g.add_edge(i, 5 + i)
        r = orient_edges(g)
        assert r.max_out_degree <= 2
        validate_orientation(g, r)
