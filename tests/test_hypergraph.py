"""Tests for the HyperGraph class."""

import numpy as np
import pytest
from chszlablib.exceptions import (
    CHSZLabLibError,
    InvalidHyperGraphError,
    GraphNotFinalizedError,
)
from chszlablib.hypergraph import HyperGraph


# ------------------------------------------------------------------
# InvalidHyperGraphError
# ------------------------------------------------------------------

class TestInvalidHyperGraphError:
    def test_is_chszlablib_error(self):
        assert issubclass(InvalidHyperGraphError, CHSZLabLibError)

    def test_is_value_error(self):
        assert issubclass(InvalidHyperGraphError, ValueError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(InvalidHyperGraphError):
            raise InvalidHyperGraphError("test message")

    def test_caught_as_value_error(self):
        with pytest.raises(ValueError):
            raise InvalidHyperGraphError("test message")


# ------------------------------------------------------------------
# Basic construction with set_edge
# ------------------------------------------------------------------

class TestSetEdge:
    def test_single_edge(self):
        """Hypergraph with one edge containing two vertices."""
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 2])
        hg.finalize()
        assert hg.num_nodes == 3
        assert hg.num_edges == 1

    def test_multiple_edges(self):
        """Hypergraph with three edges."""
        hg = HyperGraph(num_nodes=5, num_edges=3)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.set_edge(2, [3, 4])
        hg.finalize()
        assert hg.num_nodes == 5
        assert hg.num_edges == 3

    def test_set_edge_replaces_previous(self):
        """set_edge replaces any previously added vertices."""
        hg = HyperGraph(num_nodes=4, num_edges=1)
        hg.add_to_edge(0, 0)
        hg.add_to_edge(0, 1)
        hg.set_edge(0, [2, 3])
        hg.finalize()
        verts = hg.everts[hg.eptr[0]:hg.eptr[1]]
        np.testing.assert_array_equal(verts, [2, 3])


# ------------------------------------------------------------------
# Basic construction with add_to_edge
# ------------------------------------------------------------------

class TestAddToEdge:
    def test_add_vertices_one_by_one(self):
        hg = HyperGraph(num_nodes=4, num_edges=1)
        hg.add_to_edge(0, 3)
        hg.add_to_edge(0, 1)
        hg.add_to_edge(0, 0)
        hg.finalize()
        verts = hg.everts[hg.eptr[0]:hg.eptr[1]]
        # Vertices should be sorted after finalize
        np.testing.assert_array_equal(verts, [0, 1, 3])


# ------------------------------------------------------------------
# eptr / everts correctness
# ------------------------------------------------------------------

class TestEptrEverts:
    def test_eptr_shape(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.finalize()
        assert hg.eptr.shape == (3,)  # m + 1

    def test_eptr_values(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])  # 3 vertices
        hg.set_edge(1, [2, 3])     # 2 vertices
        hg.finalize()
        np.testing.assert_array_equal(hg.eptr, [0, 3, 5])

    def test_everts_values(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.finalize()
        np.testing.assert_array_equal(hg.everts, [0, 1, 2, 2, 3])

    def test_everts_sorted_within_edge(self):
        """Vertices within each edge should be sorted."""
        hg = HyperGraph(num_nodes=5, num_edges=1)
        hg.set_edge(0, [4, 2, 0, 3])
        hg.finalize()
        verts = hg.everts[hg.eptr[0]:hg.eptr[1]]
        np.testing.assert_array_equal(verts, [0, 2, 3, 4])

    def test_eptr_dtypes(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        assert hg.eptr.dtype == np.int64
        assert hg.everts.dtype == np.int32


# ------------------------------------------------------------------
# vptr / vedges correctness
# ------------------------------------------------------------------

class TestVptrVedges:
    def test_vptr_shape(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.finalize()
        assert hg.vptr.shape == (5,)  # n + 1

    def test_vptr_values(self):
        """Vertex 0 in 1 edge, vertex 1 in 1 edge, vertex 2 in 2 edges, vertex 3 in 1 edge."""
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.finalize()
        np.testing.assert_array_equal(hg.vptr, [0, 1, 2, 4, 5])

    def test_vedges_values(self):
        hg = HyperGraph(num_nodes=4, num_edges=2)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge(1, [2, 3])
        hg.finalize()
        # vertex 0 -> [edge 0]
        np.testing.assert_array_equal(hg.vedges[hg.vptr[0]:hg.vptr[1]], [0])
        # vertex 1 -> [edge 0]
        np.testing.assert_array_equal(hg.vedges[hg.vptr[1]:hg.vptr[2]], [0])
        # vertex 2 -> [edge 0, edge 1]
        np.testing.assert_array_equal(hg.vedges[hg.vptr[2]:hg.vptr[3]], [0, 1])
        # vertex 3 -> [edge 1]
        np.testing.assert_array_equal(hg.vedges[hg.vptr[3]:hg.vptr[4]], [1])

    def test_vptr_dtypes(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        assert hg.vptr.dtype == np.int64
        assert hg.vedges.dtype == np.int32

    def test_vertex_not_in_any_edge(self):
        """Vertex 1 belongs to no edges."""
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 2])
        hg.finalize()
        edges_of_1 = hg.vedges[hg.vptr[1]:hg.vptr[2]]
        assert len(edges_of_1) == 0


# ------------------------------------------------------------------
# Auto-finalize on property access
# ------------------------------------------------------------------

class TestAutoFinalize:
    def test_auto_finalize_via_eptr(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        _ = hg.eptr
        assert hg._finalized is True

    def test_auto_finalize_via_everts(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        _ = hg.everts
        assert hg._finalized is True

    def test_auto_finalize_via_vptr(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        _ = hg.vptr
        assert hg._finalized is True

    def test_auto_finalize_via_vedges(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        _ = hg.vedges
        assert hg._finalized is True

    def test_auto_finalize_via_node_weights(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        _ = hg.node_weights
        assert hg._finalized is True

    def test_auto_finalize_via_edge_weights(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        _ = hg.edge_weights
        assert hg._finalized is True


# ------------------------------------------------------------------
# Idempotent finalize
# ------------------------------------------------------------------

class TestFinalizeIdempotency:
    def test_double_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        hg.finalize()
        eptr_first = hg.eptr.copy()
        everts_first = hg.everts.copy()
        vptr_first = hg.vptr.copy()
        vedges_first = hg.vedges.copy()
        hg.finalize()  # second call should be a no-op
        np.testing.assert_array_equal(hg.eptr, eptr_first)
        np.testing.assert_array_equal(hg.everts, everts_first)
        np.testing.assert_array_equal(hg.vptr, vptr_first)
        np.testing.assert_array_equal(hg.vedges, vedges_first)


# ------------------------------------------------------------------
# Default weights
# ------------------------------------------------------------------

class TestDefaultWeights:
    def test_default_node_weights_all_ones(self):
        hg = HyperGraph(num_nodes=4, num_edges=1)
        hg.set_edge(0, [0, 1, 2, 3])
        hg.finalize()
        np.testing.assert_array_equal(hg.node_weights, [1, 1, 1, 1])

    def test_default_edge_weights_all_ones(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        hg.set_edge(1, [1, 2])
        hg.finalize()
        np.testing.assert_array_equal(hg.edge_weights, [1, 1])

    def test_weight_dtypes(self):
        hg = HyperGraph(num_nodes=2, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        assert hg.node_weights.dtype == np.int64
        assert hg.edge_weights.dtype == np.int64


# ------------------------------------------------------------------
# Custom weights
# ------------------------------------------------------------------

class TestCustomWeights:
    def test_custom_node_weights(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_node_weight(0, 10)
        hg.set_node_weight(2, 30)
        hg.set_edge(0, [0, 1, 2])
        hg.finalize()
        np.testing.assert_array_equal(hg.node_weights, [10, 1, 30])

    def test_custom_edge_weights(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        hg.set_edge(1, [1, 2])
        hg.set_edge_weight(0, 5)
        hg.set_edge_weight(1, 7)
        hg.finalize()
        np.testing.assert_array_equal(hg.edge_weights, [5, 7])

    def test_mixed_weights(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        hg.set_edge(1, [1, 2])
        hg.set_node_weight(1, 42)
        hg.set_edge_weight(1, 99)
        hg.finalize()
        np.testing.assert_array_equal(hg.node_weights, [1, 42, 1])
        np.testing.assert_array_equal(hg.edge_weights, [1, 99])


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------

class TestValidation:
    def test_negative_num_nodes(self):
        with pytest.raises(InvalidHyperGraphError, match="non-negative"):
            HyperGraph(num_nodes=-1, num_edges=0)

    def test_negative_num_edges(self):
        with pytest.raises(InvalidHyperGraphError, match="non-negative"):
            HyperGraph(num_nodes=0, num_edges=-1)

    def test_vertex_out_of_range_high(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.add_to_edge(0, 5)

    def test_vertex_out_of_range_negative(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.add_to_edge(0, -1)

    def test_edge_out_of_range_high(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.add_to_edge(5, 0)

    def test_edge_out_of_range_negative(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.add_to_edge(-1, 0)

    def test_duplicate_vertex_in_edge_via_add(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.add_to_edge(0, 1)
        with pytest.raises(InvalidHyperGraphError, match="Duplicate vertex"):
            hg.add_to_edge(0, 1)

    def test_duplicate_vertex_in_edge_via_set(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="Duplicate vertex"):
            hg.set_edge(0, [1, 2, 1])

    def test_modify_add_to_edge_after_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        with pytest.raises(GraphNotFinalizedError, match="Cannot modify"):
            hg.add_to_edge(0, 2)

    def test_modify_set_edge_after_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        with pytest.raises(GraphNotFinalizedError, match="Cannot modify"):
            hg.set_edge(0, [0, 2])

    def test_set_node_weight_after_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        with pytest.raises(GraphNotFinalizedError, match="Cannot set node weights"):
            hg.set_node_weight(0, 5)

    def test_set_edge_weight_after_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1])
        hg.finalize()
        with pytest.raises(GraphNotFinalizedError, match="Cannot set edge weights"):
            hg.set_edge_weight(0, 5)

    def test_empty_edge_at_finalize(self):
        hg = HyperGraph(num_nodes=3, num_edges=2)
        hg.set_edge(0, [0, 1])
        # Edge 1 is left empty
        with pytest.raises(InvalidHyperGraphError, match="no vertices"):
            hg.finalize()

    def test_set_edge_vertex_out_of_range(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.set_edge(0, [0, 10])

    def test_set_edge_edge_out_of_range(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.set_edge(5, [0, 1])

    def test_set_node_weight_out_of_range(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.set_node_weight(5, 10)

    def test_set_edge_weight_out_of_range(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        with pytest.raises(InvalidHyperGraphError, match="out of range"):
            hg.set_edge_weight(5, 10)


# ------------------------------------------------------------------
# __repr__
# ------------------------------------------------------------------

class TestRepr:
    def test_repr_not_finalized(self):
        hg = HyperGraph(num_nodes=5, num_edges=3)
        assert repr(hg) == "HyperGraph(n=5, m=3, finalized=False)"

    def test_repr_finalized_unweighted(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        hg.finalize()
        assert repr(hg) == "HyperGraph(n=3, m=1, weighted=False)"

    def test_repr_finalized_weighted_nodes(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        hg.set_node_weight(0, 5)
        hg.finalize()
        assert repr(hg) == "HyperGraph(n=3, m=1, weighted=True)"

    def test_repr_finalized_weighted_edges(self):
        hg = HyperGraph(num_nodes=3, num_edges=1)
        hg.set_edge(0, [0, 1, 2])
        hg.set_edge_weight(0, 5)
        hg.finalize()
        assert repr(hg) == "HyperGraph(n=3, m=1, weighted=True)"


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_nodes_zero_edges(self):
        hg = HyperGraph(num_nodes=0, num_edges=0)
        hg.finalize()
        assert hg.num_nodes == 0
        assert hg.num_edges == 0
        assert len(hg.eptr) == 1
        assert hg.eptr[0] == 0
        assert len(hg.everts) == 0
        assert len(hg.vptr) == 1
        assert hg.vptr[0] == 0
        assert len(hg.vedges) == 0
        assert len(hg.node_weights) == 0
        assert len(hg.edge_weights) == 0

    def test_single_vertex_single_edge(self):
        hg = HyperGraph(num_nodes=1, num_edges=1)
        hg.set_edge(0, [0])
        hg.finalize()
        np.testing.assert_array_equal(hg.eptr, [0, 1])
        np.testing.assert_array_equal(hg.everts, [0])
        np.testing.assert_array_equal(hg.vptr, [0, 1])
        np.testing.assert_array_equal(hg.vedges, [0])

    def test_large_hyperedge(self):
        """Edge containing all vertices."""
        n = 10
        hg = HyperGraph(num_nodes=n, num_edges=1)
        hg.set_edge(0, list(range(n)))
        hg.finalize()
        verts = hg.everts[hg.eptr[0]:hg.eptr[1]]
        np.testing.assert_array_equal(verts, list(range(n)))
        # Every vertex should be in edge 0
        for v in range(n):
            edges = hg.vedges[hg.vptr[v]:hg.vptr[v + 1]]
            np.testing.assert_array_equal(edges, [0])

    def test_num_nodes_and_num_edges_are_not_auto_finalized(self):
        """num_nodes and num_edges should work without triggering finalize."""
        hg = HyperGraph(num_nodes=5, num_edges=3)
        assert hg.num_nodes == 5
        assert hg.num_edges == 3
        assert hg._finalized is False
