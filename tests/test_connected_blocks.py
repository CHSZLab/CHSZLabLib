"""Tests for KaHIP's ultra mode and connected blocks (partition and evolutionary_partition)."""

from collections import deque

import numpy as np
import pytest

from chszlablib import Graph, Decomposition


def make_grid(rows=30, cols=30):
    """rows x cols grid graph (connected)."""
    g = Graph(num_nodes=rows * cols)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                g.add_edge(v, v + 1)
            if r + 1 < rows:
                g.add_edge(v, v + cols)
    return g


def disconnected_blocks(g, assignment):
    """Return the block IDs whose induced subgraph is not connected."""
    g.finalize()
    n = len(assignment)
    seen = np.zeros(n, dtype=bool)
    components = {}
    for s in range(n):
        if seen[s]:
            continue
        components[assignment[s]] = components.get(assignment[s], 0) + 1
        seen[s] = True
        queue = deque([s])
        while queue:
            v = queue.popleft()
            for u in g.adjncy[g.xadj[v]:g.xadj[v + 1]]:
                if not seen[u] and assignment[u] == assignment[v]:
                    seen[u] = True
                    queue.append(u)
    return [b for b, c in components.items() if c > 1]


def max_block_size(assignment, k):
    return np.bincount(assignment, minlength=k).max()


class TestUltraMode:
    def test_partition_ultra(self):
        g = make_grid()
        k = 4
        result = Decomposition.partition(g, num_parts=k, mode="ultra")
        assert len(result.assignment) == g.num_nodes
        assert set(np.unique(result.assignment)) == set(range(k))
        assert max_block_size(result.assignment, k) <= 1.03 * np.ceil(g.num_nodes / k)

    def test_ultra_not_worse_than_eco(self):
        g = make_grid()
        eco = Decomposition.partition(g, num_parts=8, mode="eco", seed=1)
        ultra = Decomposition.partition(g, num_parts=8, mode="ultra", seed=1)
        assert ultra.edgecut <= eco.edgecut * 1.1

    def test_evolutionary_ultra(self):
        g = make_grid()
        result = Decomposition.evolutionary_partition(g, num_parts=4, time_limit=1, mode="ultra")
        assert set(np.unique(result.assignment)) == set(range(4))

    def test_ultrafastsocial_still_works(self):
        g = make_grid()
        result = Decomposition.evolutionary_partition(
            g, num_parts=4, time_limit=1, mode="ultrafastsocial")
        assert set(np.unique(result.assignment)) == set(range(4))

    def test_mode_constants(self):
        from chszlablib import _kahip, _kahipe
        assert _kahip.ULTRA == 6
        assert _kahipe.ULTRA == 6
        assert _kahipe.ULTRAFASTSOCIAL == 7


class TestConnectedBlocks:
    @pytest.mark.parametrize("k", [2, 4, 8, 16])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_partition_connected(self, k, seed):
        g = make_grid()
        result = Decomposition.partition(
            g, num_parts=k, mode="strong", seed=seed, connected_blocks=True)
        assert disconnected_blocks(g, result.assignment) == []
        assert max_block_size(result.assignment, k) <= 1.03 * np.ceil(g.num_nodes / k)
        assert result.edgecut > 0

    @pytest.mark.parametrize("k", [4, 8])
    def test_evolutionary_connected(self, k):
        g = make_grid()
        result = Decomposition.evolutionary_partition(
            g, num_parts=k, time_limit=1, mode="strong", connected_blocks=True)
        assert disconnected_blocks(g, result.assignment) == []
        assert max_block_size(result.assignment, k) <= 1.03 * np.ceil(g.num_nodes / k)

    def test_default_is_unchanged(self):
        g = make_grid()
        a = Decomposition.partition(g, num_parts=4, mode="strong", seed=3)
        b = Decomposition.partition(g, num_parts=4, mode="strong", seed=3, connected_blocks=False)
        assert a.edgecut == b.edgecut
        assert np.array_equal(a.assignment, b.assignment)

    @pytest.mark.parametrize("mode", ["fast", "eco", "ultra", "strongsocial"])
    def test_requires_strong(self, mode):
        g = make_grid(5, 5)
        with pytest.raises(ValueError, match="connected_blocks"):
            Decomposition.partition(g, num_parts=2, mode=mode, connected_blocks=True)
        with pytest.raises(ValueError, match="connected_blocks"):
            Decomposition.evolutionary_partition(
                g, num_parts=2, time_limit=1, mode=mode, connected_blocks=True)

    def test_uppercase_strong_accepted(self):
        g = make_grid(10, 10)
        result = Decomposition.partition(g, num_parts=2, mode="STRONG", connected_blocks=True)
        assert disconnected_blocks(g, result.assignment) == []
