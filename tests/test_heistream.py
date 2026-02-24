"""Tests for HeiStream streaming graph partitioner."""

import numpy as np
import pytest

from chszlablib import Graph, Decomposition
from chszlablib.heistream import HeiStreamPartitioner, StreamPartitionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_path_graph(n: int) -> Graph:
    """0 -- 1 -- 2 -- ... -- (n-1)"""
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def make_cycle_graph(n: int) -> Graph:
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    g.add_edge(n - 1, 0)
    return g


def make_grid_graph(rows: int, cols: int) -> Graph:
    """rows x cols grid."""
    n = rows * cols
    g = Graph(n)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                g.add_edge(v, v + 1)
            if r + 1 < rows:
                g.add_edge(v, v + cols)
    return g


# ---------------------------------------------------------------------------
# Tests – stream_partition (Graph-based convenience)
# ---------------------------------------------------------------------------

class TestStreamPartition:
    """Tests using the stream_partition convenience function."""

    def test_path_2_parts(self):
        g = make_path_graph(10)
        result = Decomposition.stream_partition(g, k=2)
        assert isinstance(result, StreamPartitionResult)
        assert result.assignment.shape == (10,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_cycle_4_parts(self):
        g = make_cycle_graph(20)
        result = Decomposition.stream_partition(g, k=4)
        assert result.assignment.shape == (20,)
        assert set(np.unique(result.assignment)) <= set(range(4))

    def test_grid_2_parts(self):
        g = make_grid_graph(5, 6)
        result = Decomposition.stream_partition(g, k=2)
        assert result.assignment.shape == (30,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_larger_graph(self):
        g = make_grid_graph(10, 10)
        result = Decomposition.stream_partition(g, k=4, imbalance=5.0)
        assert result.assignment.shape == (100,)
        parts = np.unique(result.assignment)
        assert len(parts) <= 4
        # Check rough balance: no part should have more than (1 + eps) * n/k
        for p in parts:
            assert np.sum(result.assignment == p) <= 100 * (1 + 5.0 / 100) / 4 + 2

    def test_seed_determinism(self):
        g = make_grid_graph(6, 6)
        r1 = Decomposition.stream_partition(g, k=3, seed=42)
        r2 = Decomposition.stream_partition(g, k=3, seed=42)
        np.testing.assert_array_equal(r1.assignment, r2.assignment)


# ---------------------------------------------------------------------------
# Tests – HeiStreamPartitioner (streaming node-by-node API)
# ---------------------------------------------------------------------------

class TestHeiStreamPartitioner:
    """Tests using the incremental node-by-node API."""

    def test_basic_streaming(self):
        hs = HeiStreamPartitioner(k=2)
        # Small path: 0-1-2-3
        hs.new_node(0, [1])
        hs.new_node(1, [0, 2])
        hs.new_node(2, [1, 3])
        hs.new_node(3, [2])
        result = hs.partition()
        assert result.assignment.shape == (4,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_duplicate_node_raises(self):
        hs = HeiStreamPartitioner(k=2)
        hs.new_node(0, [1])
        with pytest.raises(ValueError, match="already been added"):
            hs.new_node(0, [1])

    def test_empty_graph(self):
        hs = HeiStreamPartitioner(k=2)
        result = hs.partition()
        assert result.assignment.shape == (0,)

    def test_reset(self):
        hs = HeiStreamPartitioner(k=2)
        hs.new_node(0, [1])
        hs.new_node(1, [0])
        hs.reset()
        # After reset, can add nodes again
        hs.new_node(0, [1, 2])
        hs.new_node(1, [0])
        hs.new_node(2, [0])
        result = hs.partition()
        assert result.assignment.shape == (3,)

    def test_streaming_matches_graph(self):
        """Streaming API should produce same result as graph-based API."""
        g = make_path_graph(8)
        r_graph = Decomposition.stream_partition(g, k=2, seed=123)

        hs = HeiStreamPartitioner(k=2, seed=123)
        g.finalize()
        for v in range(g.num_nodes):
            start, end = g.xadj[v], g.xadj[v + 1]
            neighbors = g.adjncy[start:end].tolist()
            hs.new_node(v, neighbors)
        r_stream = hs.partition()

        np.testing.assert_array_equal(r_graph.assignment, r_stream.assignment)


# ---------------------------------------------------------------------------
# Tests – BuffCut (buffer > 1)
# ---------------------------------------------------------------------------

class TestBuffCut:
    """Tests with explicit buffer sizes to exercise BuffCut mode."""

    def test_buffcut_small_buffer(self):
        g = make_grid_graph(6, 6)
        result = Decomposition.stream_partition(g, k=2, max_buffer_size=10, batch_size=4)
        assert result.assignment.shape == (36,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_buffcut_large_buffer(self):
        g = make_grid_graph(8, 8)
        result = Decomposition.stream_partition(g, k=4, max_buffer_size=50, batch_size=8)
        assert result.assignment.shape == (64,)
        assert set(np.unique(result.assignment)) <= set(range(4))

    def test_direct_fennel(self):
        """buffer_size=1 forces direct Fennel one-pass."""
        g = make_grid_graph(6, 6)
        result = Decomposition.stream_partition(g, k=2, max_buffer_size=1)
        assert result.assignment.shape == (36,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_restreaming(self):
        g = make_grid_graph(6, 6)
        result = Decomposition.stream_partition(g, k=2, num_streams_passes=2, max_buffer_size=10, batch_size=4)
        assert result.assignment.shape == (36,)
        assert set(np.unique(result.assignment)) <= {0, 1}
