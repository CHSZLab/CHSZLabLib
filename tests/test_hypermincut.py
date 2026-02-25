"""Tests for Decomposition.hypermincut() -- HeiCut integration."""

import pytest

from chszlablib import HyperGraph, Decomposition, HyperMincutResult
from chszlablib.exceptions import InvalidModeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hg(edges, weights=None, num_nodes=None):
    """Build a HyperGraph from an edge list with optional weights."""
    return HyperGraph.from_edge_list(edges, num_nodes=num_nodes, edge_weights=weights)


def _gurobi_available() -> bool:
    """Return True if gurobipy is installed AND the license is valid."""
    try:
        import gurobipy as gp
        m = gp.Model("_test")
        m.dispose()
        del m
        return True
    except Exception:
        return False


_has_gurobi = _gurobi_available()


# ---------------------------------------------------------------------------
# Kernelizer (default method)
# ---------------------------------------------------------------------------

class TestHypermincutKernelizer:

    def test_single_edge(self):
        hg = _make_hg([[0, 1]])
        r = Decomposition.hypermincut(hg)
        assert isinstance(r, HyperMincutResult)
        assert r.cut_value == 1
        assert r.method == "kernelizer"
        assert r.time >= 0.0

    def test_two_disjoint_edges(self):
        hg = _make_hg([[0, 1], [2, 3]])
        r = Decomposition.hypermincut(hg)
        assert r.cut_value == 0

    def test_path_like(self):
        hg = _make_hg([[0, 1], [1, 2], [2, 3]])
        r = Decomposition.hypermincut(hg)
        assert r.cut_value == 1

    def test_two_edges_sharing_vertex(self):
        hg = _make_hg([[0, 1, 2], [2, 3, 4]])
        r = Decomposition.hypermincut(hg)
        assert r.cut_value == 1

    def test_weighted_edges(self):
        hg = _make_hg([[0, 1], [1, 2]], weights=[3, 5])
        r = Decomposition.hypermincut(hg)
        assert r.cut_value == 3

    def test_triangle_hyperedge(self):
        hg = _make_hg([[0, 1, 2]])
        r = Decomposition.hypermincut(hg)
        assert r.cut_value == 1


# ---------------------------------------------------------------------------
# Submodular
# ---------------------------------------------------------------------------

class TestHypermincutSubmodular:

    def test_single_edge(self):
        hg = _make_hg([[0, 1]])
        r = Decomposition.hypermincut(hg, method="submodular")
        assert r.cut_value == 1
        assert r.method == "submodular"

    def test_path_like(self):
        hg = _make_hg([[0, 1], [1, 2], [2, 3]])
        r = Decomposition.hypermincut(hg, method="submodular")
        assert r.cut_value == 1


# ---------------------------------------------------------------------------
# Trimmer
# ---------------------------------------------------------------------------

class TestHypermincutTrimmer:

    def test_single_edge(self):
        hg = _make_hg([[0, 1]])
        r = Decomposition.hypermincut(hg, method="trimmer")
        assert r.cut_value == 1
        assert r.method == "trimmer"

    def test_path_like(self):
        hg = _make_hg([[0, 1], [1, 2], [2, 3]])
        r = Decomposition.hypermincut(hg, method="trimmer")
        assert r.cut_value == 1


# ---------------------------------------------------------------------------
# ILP
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_gurobi, reason="gurobipy not available or license expired")
class TestHypermincutILP:

    def test_ilp_single_edge(self):
        """ILP should find the correct minimum cut."""
        hg = _make_hg([[0, 1]])
        r = Decomposition.hypermincut(hg, method="ilp", time_limit=30.0)
        assert r.cut_value == 1
        assert r.method == "ilp"

    def test_ilp_path_like(self):
        hg = _make_hg([[0, 1], [1, 2], [2, 3]])
        r = Decomposition.hypermincut(hg, method="ilp", time_limit=30.0)
        assert r.cut_value == 1

    def test_ilp_disjoint(self):
        hg = _make_hg([[0, 1], [2, 3]])
        r = Decomposition.hypermincut(hg, method="ilp", time_limit=30.0)
        assert r.cut_value == 0

    def test_ilp_weighted(self):
        hg = _make_hg([[0, 1], [1, 2]], weights=[3, 5])
        r = Decomposition.hypermincut(hg, method="ilp", time_limit=30.0)
        assert r.cut_value == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestHypermincutValidation:

    def test_invalid_method(self):
        hg = _make_hg([[0, 1]])
        with pytest.raises(InvalidModeError, match="Unknown method"):
            Decomposition.hypermincut(hg, method="bogus")


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestHypermincutIntrospection:

    def test_in_available_methods(self):
        methods = Decomposition.available_methods()
        assert "hypermincut" in methods

    def test_methods_tuple(self):
        assert "kernelizer" in Decomposition.HYPERMINCUT_METHODS
        assert "submodular" in Decomposition.HYPERMINCUT_METHODS
        assert "ilp" in Decomposition.HYPERMINCUT_METHODS
        assert "trimmer" in Decomposition.HYPERMINCUT_METHODS
