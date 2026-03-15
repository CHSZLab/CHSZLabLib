"""Tests for the Graph class."""

import numpy as np
import pytest
from chszlablib import Graph


# ------------------------------------------------------------------
# Empty graph
# ------------------------------------------------------------------

class TestEmptyGraph:
    def test_empty_graph_properties(self):
        g = Graph(num_nodes=0)
        g.finalize()
        assert g.num_nodes == 0
        assert g.num_edges == 0
        assert len(g.xadj) == 1
        assert g.xadj[0] == 0
        assert len(g.adjncy) == 0
        assert len(g.node_weights) == 0
        assert len(g.edge_weights) == 0

    def test_nodes_no_edges(self):
        g = Graph(num_nodes=5)
        g.finalize()
        assert g.num_nodes == 5
        assert g.num_edges == 0
        np.testing.assert_array_equal(g.xadj, [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(g.node_weights, [1, 1, 1, 1, 1])


# ------------------------------------------------------------------
# Basic edge addition
# ------------------------------------------------------------------

class TestAddEdge:
    def test_single_edge(self):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1)
        g.finalize()
        assert g.num_edges == 1
        # Both directions stored
        assert len(g.adjncy) == 2

    def test_both_directions_stored(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 2)
        g.finalize()
        # Node 0 should have neighbor 2
        neighbors_0 = g.adjncy[g.xadj[0]:g.xadj[1]]
        assert 2 in neighbors_0
        # Node 2 should have neighbor 0
        neighbors_2 = g.adjncy[g.xadj[2]:g.xadj[3]]
        assert 0 in neighbors_2

    def test_multiple_edges(self):
        g = Graph(num_nodes=4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.finalize()
        assert g.num_edges == 3
        assert len(g.adjncy) == 6

    def test_adjacency_sorted(self):
        """Adjacency lists should be sorted by neighbor index."""
        g = Graph(num_nodes=4)
        g.add_edge(0, 3)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.finalize()
        neighbors_0 = g.adjncy[g.xadj[0]:g.xadj[1]]
        np.testing.assert_array_equal(neighbors_0, [1, 2, 3])


# ------------------------------------------------------------------
# Weights
# ------------------------------------------------------------------

class TestWeights:
    def test_default_edge_weights(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.finalize()
        np.testing.assert_array_equal(g.edge_weights, [1, 1, 1, 1])

    def test_custom_edge_weights(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1, weight=5)
        g.add_edge(1, 2, weight=3)
        g.finalize()
        # Node 0 -> 1 (weight 5), Node 1 -> [0, 2] (weights [5, 3]), Node 2 -> 1 (weight 3)
        idx_0 = slice(int(g.xadj[0]), int(g.xadj[1]))
        idx_1 = slice(int(g.xadj[1]), int(g.xadj[2]))
        idx_2 = slice(int(g.xadj[2]), int(g.xadj[3]))
        assert g.edge_weights[idx_0].tolist() == [5]
        assert g.edge_weights[idx_1].tolist() == [5, 3]
        assert g.edge_weights[idx_2].tolist() == [3]

    def test_default_node_weights(self):
        g = Graph(num_nodes=3)
        g.finalize()
        np.testing.assert_array_equal(g.node_weights, [1, 1, 1])

    def test_custom_node_weights(self):
        g = Graph(num_nodes=3)
        g.set_node_weight(0, 10)
        g.set_node_weight(2, 20)
        g.finalize()
        np.testing.assert_array_equal(g.node_weights, [10, 1, 20])


# ------------------------------------------------------------------
# from_csr
# ------------------------------------------------------------------

class TestFromCSR:
    def test_from_csr_basic(self):
        # 0 -- 1 -- 2
        xadj = np.array([0, 1, 3, 4], dtype=np.int64)
        adjncy = np.array([1, 0, 2, 1], dtype=np.int32)
        g = Graph.from_csr(xadj, adjncy)
        assert g.num_nodes == 3
        assert g.num_edges == 2
        np.testing.assert_array_equal(g.xadj, xadj)
        np.testing.assert_array_equal(g.adjncy, adjncy)

    def test_from_csr_default_weights(self):
        xadj = np.array([0, 1, 2], dtype=np.int64)
        adjncy = np.array([1, 0], dtype=np.int32)
        g = Graph.from_csr(xadj, adjncy)
        np.testing.assert_array_equal(g.node_weights, [1, 1])
        np.testing.assert_array_equal(g.edge_weights, [1, 1])

    def test_from_csr_custom_weights(self):
        xadj = np.array([0, 1, 2], dtype=np.int64)
        adjncy = np.array([1, 0], dtype=np.int32)
        nw = np.array([10, 20], dtype=np.int64)
        ew = np.array([5, 5], dtype=np.int64)
        g = Graph.from_csr(xadj, adjncy, node_weights=nw, edge_weights=ew)
        np.testing.assert_array_equal(g.node_weights, [10, 20])
        np.testing.assert_array_equal(g.edge_weights, [5, 5])

    def test_from_csr_is_finalized(self):
        xadj = np.array([0, 1, 2], dtype=np.int64)
        adjncy = np.array([1, 0], dtype=np.int32)
        g = Graph.from_csr(xadj, adjncy)
        # Should be finalized already -- adding an edge should fail
        with pytest.raises(RuntimeError):
            g.add_edge(0, 1)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

class TestValidation:
    def test_node_out_of_range_high(self):
        g = Graph(num_nodes=3)
        with pytest.raises(ValueError, match="out of range"):
            g.add_edge(0, 5)

    def test_node_out_of_range_negative(self):
        g = Graph(num_nodes=3)
        with pytest.raises(ValueError, match="out of range"):
            g.add_edge(-1, 0)

    def test_self_loop(self):
        g = Graph(num_nodes=3)
        with pytest.raises(ValueError, match="Self-loop"):
            g.add_edge(1, 1)

    def test_duplicate_edge(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        with pytest.raises(ValueError, match="Duplicate"):
            g.add_edge(0, 1)

    def test_duplicate_edge_reversed(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        with pytest.raises(ValueError, match="Duplicate"):
            g.add_edge(1, 0)

    def test_add_edge_after_finalize(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.finalize()
        with pytest.raises(RuntimeError, match="Cannot add edges"):
            g.add_edge(1, 2)

    def test_set_node_weight_after_finalize(self):
        g = Graph(num_nodes=3)
        g.finalize()
        with pytest.raises(RuntimeError, match="Cannot set node weights"):
            g.set_node_weight(0, 5)

    def test_set_node_weight_out_of_range(self):
        g = Graph(num_nodes=3)
        with pytest.raises(ValueError, match="out of range"):
            g.set_node_weight(5, 10)

    def test_negative_num_nodes(self):
        with pytest.raises(ValueError, match="non-negative"):
            Graph(num_nodes=-1)


# ------------------------------------------------------------------
# CSR structure correctness
# ------------------------------------------------------------------

class TestCSRStructure:
    def test_xadj_shape(self):
        g = Graph(num_nodes=5)
        g.add_edge(0, 1)
        g.finalize()
        assert g.xadj.shape == (6,)  # n + 1

    def test_xadj_monotonic(self):
        g = Graph(num_nodes=5)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.finalize()
        for i in range(g.num_nodes):
            assert g.xadj[i + 1] >= g.xadj[i]

    def test_adjncy_length(self):
        g = Graph(num_nodes=4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.finalize()
        # 3 undirected edges -> 6 directed entries
        assert len(g.adjncy) == 6
        assert len(g.edge_weights) == 6

    def test_xadj_last_equals_adjncy_length(self):
        g = Graph(num_nodes=4)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        g.finalize()
        assert g.xadj[-1] == len(g.adjncy)

    def test_dtypes(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.finalize()
        assert g.xadj.dtype == np.int64
        assert g.adjncy.dtype == np.int32
        assert g.node_weights.dtype == np.int64
        assert g.edge_weights.dtype == np.int64


# ------------------------------------------------------------------
# Finalize idempotency
# ------------------------------------------------------------------

class TestFinalizeIdempotency:
    def test_double_finalize(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.finalize()
        xadj_first = g.xadj.copy()
        adjncy_first = g.adjncy.copy()
        g.finalize()  # second call should be a no-op
        np.testing.assert_array_equal(g.xadj, xadj_first)
        np.testing.assert_array_equal(g.adjncy, adjncy_first)

    def test_auto_finalize_via_property(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        # Accessing property should auto-finalize
        _ = g.xadj
        assert g._finalized is True
        assert g.num_edges == 1


# ------------------------------------------------------------------
# conftest fixtures
# ------------------------------------------------------------------

class TestFixtures:
    def test_simple_path_graph(self, simple_path_graph):
        g = simple_path_graph
        assert g.num_nodes == 4
        assert g.num_edges == 3

    def test_weighted_graph(self, weighted_graph):
        g = weighted_graph
        assert g.num_nodes == 5
        assert g.num_edges == 6


# ------------------------------------------------------------------
# repr
# ------------------------------------------------------------------

class TestRepr:
    def test_repr(self):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        assert repr(g) == "Graph(n=3, edges_added=2, finalized=False)"
        g.finalize()
        assert repr(g) == "Graph(n=3, m=2, weighted=False)"

    def test_repr_weighted(self):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1, weight=5)
        g.finalize()
        assert repr(g) == "Graph(n=2, m=1, weighted=True)"
