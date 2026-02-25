"""Tests for IndependenceProblems.hypermis() — HyperMIS integration."""

import numpy as np
import pytest

from chszlablib import HyperGraph, IndependenceProblems, HyperMISResult
from chszlablib.exceptions import InvalidModeError

_has_gurobipy = IndependenceProblems.HYPERMIS_ILP_AVAILABLE


class TestHyperMISResult:
    """Verify the HyperMISResult dataclass."""

    def test_fields(self):
        r = HyperMISResult(size=2, weight=5, vertices=np.array([0, 3]),
                           offset=2, reduction_time=0.01, is_optimal=False)
        assert r.size == 2
        assert r.weight == 5
        assert r.offset == 2
        assert r.reduction_time == 0.01
        assert r.is_optimal is False
        np.testing.assert_array_equal(r.vertices, [0, 3])


def _verify_independence(hg, result):
    """Assert that the selected vertices form a valid independent set."""
    selected = set(result.vertices.tolist())
    for eid in range(hg.num_edges):
        start, end = hg.eptr[eid], hg.eptr[eid + 1]
        edge_verts = set(hg.everts[start:end].tolist())
        assert len(selected & edge_verts) <= 1, (
            f"Edge {eid} has >1 selected vertex: {selected & edge_verts}"
        )


class TestHypermisHeuristic:
    """Test method='heuristic' (default)."""

    def test_simple_two_edge_hypergraph(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert isinstance(result, HyperMISResult)
        assert result.size >= 1
        assert len(result.vertices) == result.size
        assert result.reduction_time >= 0.0
        assert result.is_optimal is False
        _verify_independence(hg, result)

    def test_disjoint_edges(self):
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert result.size >= 2
        _verify_independence(hg, result)

    def test_single_large_edge(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2, 3, 4]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert result.size == 1
        assert len(result.vertices) == 1

    def test_strong_reductions_off(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0,
                                               strong_reductions=False)
        assert isinstance(result, HyperMISResult)
        assert result.size >= 1

    def test_weight_computation_with_weights(self):
        hg = HyperGraph.from_edge_list(
            [[0, 1], [2, 3]],
            node_weights=np.array([10, 20, 30, 40]),
        )
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        expected_weight = int(np.sum(hg.node_weights[result.vertices]))
        assert result.weight == expected_weight

    def test_weight_defaults_to_size_without_weights(self):
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert result.weight == result.size

    def test_negative_time_limit(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        with pytest.raises(ValueError, match="time_limit must be >= 0"):
            IndependenceProblems.hypermis(hg, time_limit=-1.0)

    def test_offset_nonnegative(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3]])
        result = IndependenceProblems.hypermis(hg, time_limit=5.0)
        assert result.offset >= 0


class TestHypermisExact:
    """Test method='exact' (ILP path)."""

    def test_exact_without_gurobi(self, monkeypatch):
        """method='exact' raises ImportError when gurobipy is not available."""
        import chszlablib.independence as mod
        monkeypatch.setattr(mod, "_HYPERMIS_ILP_AVAILABLE", False)
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3]])
        with pytest.raises(ImportError, match="gurobipy"):
            IndependenceProblems.hypermis(hg, method="exact")

    @pytest.mark.skipif(not _has_gurobipy, reason="gurobipy not installed")
    def test_exact_small_hypergraph(self):
        hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3], [4, 5]])
        result = IndependenceProblems.hypermis(hg, method="exact", time_limit=10.0)
        assert isinstance(result, HyperMISResult)
        assert result.size >= 1
        assert len(result.vertices) == result.size
        _verify_independence(hg, result)

    @pytest.mark.skipif(not _has_gurobipy, reason="gurobipy not installed")
    def test_exact_is_optimal(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        result = IndependenceProblems.hypermis(hg, method="exact", time_limit=10.0)
        assert result.is_optimal is True
        assert result.size == 1

    @pytest.mark.skipif(not _has_gurobipy, reason="gurobipy not installed")
    def test_exact_disjoint_edges(self):
        hg = HyperGraph.from_edge_list([[0, 1], [2, 3], [4, 5]])
        result = IndependenceProblems.hypermis(hg, method="exact", time_limit=10.0)
        assert result.size >= 3
        _verify_independence(hg, result)


class TestHypermisValidation:
    """Test error handling."""

    def test_invalid_method(self):
        hg = HyperGraph.from_edge_list([[0, 1]])
        with pytest.raises(InvalidModeError, match="Unknown method"):
            IndependenceProblems.hypermis(hg, method="bogus")


class TestAvailableMethods:
    """Verify hypermis appears in available_methods."""

    def test_hypermis_in_methods(self):
        methods = IndependenceProblems.available_methods()
        assert "hypermis" in methods

    def test_hypermis_methods_tuple(self):
        assert "heuristic" in IndependenceProblems.HYPERMIS_METHODS
        assert "exact" in IndependenceProblems.HYPERMIS_METHODS


class TestHyperMISILPAvailable:
    """Check the HYPERMIS_ILP_AVAILABLE attribute."""

    def test_attribute_exists(self):
        assert hasattr(IndependenceProblems, "HYPERMIS_ILP_AVAILABLE")
        assert isinstance(IndependenceProblems.HYPERMIS_ILP_AVAILABLE, bool)
