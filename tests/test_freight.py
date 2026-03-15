"""Tests for FREIGHT streaming hypergraph partitioning."""

import numpy as np
import pytest

from chszlablib import (
    Decomposition,
    FreightPartitioner,
    HyperGraph,
    StreamHypergraphPartitionResult,
)


def make_small_hypergraph():
    """6 nodes, 3 nets: {0,1,2}, {2,3,4}, {4,5,0}."""
    return HyperGraph.from_edge_list(
        [[0, 1, 2], [2, 3, 4], [4, 5, 0]], num_nodes=6
    )


class TestStreamHypergraphPartition:
    """Tests for Decomposition.stream_hypergraph_partition (one-shot)."""

    def test_basic_partition(self):
        hg = make_small_hypergraph()
        result = Decomposition.stream_hypergraph_partition(hg, k=2)
        assert isinstance(result, StreamHypergraphPartitionResult)
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_determinism(self):
        hg = make_small_hypergraph()
        r1 = Decomposition.stream_hypergraph_partition(hg, k=2, seed=42)
        r2 = Decomposition.stream_hypergraph_partition(hg, k=2, seed=42)
        np.testing.assert_array_equal(r1.assignment, r2.assignment)

    def test_different_seeds(self):
        """Different seeds may produce different results (not guaranteed)."""
        hg = make_small_hypergraph()
        r1 = Decomposition.stream_hypergraph_partition(hg, k=2, seed=0)
        r2 = Decomposition.stream_hypergraph_partition(hg, k=2, seed=999)
        # Just check both are valid
        assert set(np.unique(r1.assignment)) <= {0, 1}
        assert set(np.unique(r2.assignment)) <= {0, 1}

    @pytest.mark.parametrize("algo", [
        "fennel_approx_sqrt", "fennel", "ldg", "hashing",
    ])
    def test_all_algorithms(self, algo):
        hg = make_small_hypergraph()
        result = Decomposition.stream_hypergraph_partition(
            hg, k=2, algorithm=algo, seed=0
        )
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    @pytest.mark.parametrize("objective", ["cut_net", "connectivity"])
    def test_both_objectives(self, objective):
        hg = make_small_hypergraph()
        result = Decomposition.stream_hypergraph_partition(
            hg, k=2, objective=objective, seed=0
        )
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_k3(self):
        hg = make_small_hypergraph()
        result = Decomposition.stream_hypergraph_partition(hg, k=3, seed=0)
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= {0, 1, 2}

    def test_weighted_nodes(self):
        hg = HyperGraph.from_edge_list(
            [[0, 1, 2], [2, 3, 4], [4, 5, 0]],
            num_nodes=6,
            node_weights=[10, 1, 1, 1, 1, 10],
        )
        result = Decomposition.stream_hypergraph_partition(hg, k=2, seed=0)
        assert result.assignment.shape == (6,)

    def test_weighted_edges(self):
        hg = HyperGraph.from_edge_list(
            [[0, 1, 2], [2, 3, 4], [4, 5, 0]],
            num_nodes=6,
            edge_weights=[10, 1, 1],
        )
        result = Decomposition.stream_hypergraph_partition(hg, k=2, seed=0)
        assert result.assignment.shape == (6,)

    def test_balance(self):
        """Check that no block exceeds the imbalance constraint."""
        n = 100
        nets = [[i, (i + 1) % n] for i in range(n)]
        hg = HyperGraph.from_edge_list(nets, num_nodes=n)
        k = 4
        imbalance = 3.0
        result = Decomposition.stream_hypergraph_partition(
            hg, k=k, imbalance=imbalance, seed=0
        )
        max_allowed = int(np.ceil((100 + imbalance) / 100.0 * n / k))
        counts = np.bincount(result.assignment, minlength=k)
        assert np.all(counts <= max_allowed), (
            f"Block sizes {counts} exceed max_allowed={max_allowed}"
        )

    def test_hierarchical(self):
        hg = make_small_hypergraph()
        result = Decomposition.stream_hypergraph_partition(
            hg, k=4, hierarchical=True, seed=0
        )
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= set(range(4))

    def test_multi_pass(self):
        hg = make_small_hypergraph()
        result = Decomposition.stream_hypergraph_partition(
            hg, k=2, num_streams_passes=3, seed=0
        )
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_invalid_k(self):
        hg = make_small_hypergraph()
        with pytest.raises(ValueError, match="k must be >= 2"):
            Decomposition.stream_hypergraph_partition(hg, k=1)

    def test_invalid_algorithm(self):
        hg = make_small_hypergraph()
        with pytest.raises(RuntimeError, match="unknown algorithm"):
            Decomposition.stream_hypergraph_partition(hg, k=2, algorithm="bad")

    def test_invalid_objective(self):
        hg = make_small_hypergraph()
        with pytest.raises(RuntimeError, match="unknown objective"):
            Decomposition.stream_hypergraph_partition(hg, k=2, objective="bad")


class TestFreightPartitioner:
    """Tests for FreightPartitioner (true streaming)."""

    def test_basic_streaming(self):
        fp = FreightPartitioner(num_nodes=6, num_nets=3, k=2, seed=0)
        blocks = []
        blocks.append(fp.assign_node(0, nets=[[0, 1, 2], [0, 4, 5]]))
        blocks.append(fp.assign_node(1, nets=[[0, 1, 2]]))
        blocks.append(fp.assign_node(2, nets=[[0, 1, 2], [2, 3, 4]]))
        blocks.append(fp.assign_node(3, nets=[[2, 3, 4]]))
        blocks.append(fp.assign_node(4, nets=[[2, 3, 4], [0, 4, 5]]))
        blocks.append(fp.assign_node(5, nets=[[0, 4, 5]]))

        result = fp.get_assignment()
        assert isinstance(result, StreamHypergraphPartitionResult)
        assert result.assignment.shape == (6,)
        assert set(np.unique(result.assignment)) <= {0, 1}

        # Check that returned blocks match the assignment
        for i, b in enumerate(blocks):
            assert result.assignment[i] == b

    def test_immediate_return(self):
        """Each assign_node call returns a valid block ID."""
        fp = FreightPartitioner(num_nodes=4, num_nets=2, k=2, seed=0)
        b0 = fp.assign_node(0, nets=[[0, 1], [0, 2]])
        assert isinstance(b0, int)
        assert 0 <= b0 < 2

    def test_net_deduplication(self):
        """Same net vertices from different nodes map to same internal net."""
        fp = FreightPartitioner(num_nodes=3, num_nets=1, k=2, seed=0)
        fp.assign_node(0, nets=[[0, 1, 2]])
        fp.assign_node(1, nets=[[0, 1, 2]])  # same net
        fp.assign_node(2, nets=[[0, 1, 2]])  # same net
        result = fp.get_assignment()
        assert result.assignment.shape == (3,)

    def test_net_order_invariance(self):
        """Net [0,2,1] should be recognized as same net as [0,1,2]."""
        fp = FreightPartitioner(num_nodes=3, num_nets=1, k=2, seed=0)
        fp.assign_node(0, nets=[[0, 2, 1]])  # unsorted
        fp.assign_node(1, nets=[[1, 0, 2]])  # different order
        fp.assign_node(2, nets=[[0, 1, 2]])  # canonical order
        result = fp.get_assignment()
        assert result.assignment.shape == (3,)

    def test_with_weights(self):
        fp = FreightPartitioner(num_nodes=4, num_nets=2, k=2, seed=0)
        fp.assign_node(0, nets=[[0, 1], [0, 2]], net_weights=[5, 1], node_weight=2)
        fp.assign_node(1, nets=[[0, 1]], net_weights=[5], node_weight=1)
        fp.assign_node(2, nets=[[0, 2]], net_weights=[1], node_weight=1)
        fp.assign_node(3, nets=[[0, 2]], net_weights=[1], node_weight=2)
        result = fp.get_assignment()
        assert result.assignment.shape == (4,)

    @pytest.mark.parametrize("algo", [
        "fennel_approx_sqrt", "fennel", "ldg", "hashing",
    ])
    def test_streaming_all_algorithms(self, algo):
        fp = FreightPartitioner(
            num_nodes=4, num_nets=2, k=2, algorithm=algo, seed=0
        )
        fp.assign_node(0, nets=[[0, 1], [0, 3]])
        fp.assign_node(1, nets=[[0, 1]])
        fp.assign_node(2, nets=[[0, 1]])
        fp.assign_node(3, nets=[[0, 3]])
        result = fp.get_assignment()
        assert result.assignment.shape == (4,)
        assert set(np.unique(result.assignment)) <= {0, 1}

    def test_too_many_nets_raises(self):
        fp = FreightPartitioner(num_nodes=3, num_nets=1, k=2, seed=0)
        fp.assign_node(0, nets=[[0, 1]])
        with pytest.raises(RuntimeError, match="more nets than num_nets"):
            fp.assign_node(1, nets=[[1, 2]])  # new net, but num_nets=1

    def test_invalid_node_id(self):
        fp = FreightPartitioner(num_nodes=3, num_nets=1, k=2, seed=0)
        with pytest.raises(RuntimeError, match="node_id out of range"):
            fp.assign_node(5, nets=[[0, 1]])
