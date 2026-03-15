"""Tests for METIS I/O (read_metis / write_metis) and binary I/O."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from chszlablib import Graph, HyperGraph, read_metis, write_metis
from chszlablib.io import _read_metis_python, _read_hmetis_python


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_tmp(content: str) -> Path:
    """Write *content* to a temporary file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".graph", delete=False
    )
    f.write(content)
    f.close()
    return Path(f.name)


# ------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------

class TestReadMetis:
    def test_unweighted(self):
        """Unweighted graph: 4 nodes, 4 edges (a square 0-1-2-3-0)."""
        content = (
            "4 4\n"
            "2 4\n"
            "1 3\n"
            "2 4\n"
            "1 3\n"
        )
        path = _write_tmp(content)
        g = read_metis(path)
        assert g.num_nodes == 4
        assert g.num_edges == 4
        # All weights should be 1
        np.testing.assert_array_equal(g.node_weights, [1, 1, 1, 1])
        np.testing.assert_array_equal(g.edge_weights, np.ones(8, dtype=np.int64))

    def test_edge_weights_only(self):
        """fmt=1: edge weights only."""
        content = (
            "3 2 1\n"
            "2 5\n"
            "1 5 3 3\n"
            "2 3\n"
        )
        # 0--1 (w=5), 1--2 (w=3)
        path = _write_tmp(content)
        g = read_metis(path)
        assert g.num_nodes == 3
        assert g.num_edges == 2
        # Check node 1 neighbors: 0(w=5) and 2(w=3)
        start, end = int(g.xadj[1]), int(g.xadj[2])
        neighbors = g.adjncy[start:end].tolist()
        weights = g.edge_weights[start:end].tolist()
        assert set(zip(neighbors, weights)) == {(0, 5), (2, 3)}

    def test_node_weights_only(self):
        """fmt=10: node weights only."""
        content = (
            "3 2 10\n"
            "10 2\n"
            "20 1 3\n"
            "30 2\n"
        )
        path = _write_tmp(content)
        g = read_metis(path)
        assert g.num_nodes == 3
        assert g.num_edges == 2
        np.testing.assert_array_equal(g.node_weights, [10, 20, 30])
        # Edge weights should default to 1
        np.testing.assert_array_equal(g.edge_weights, np.ones(4, dtype=np.int64))

    def test_both_weights(self):
        """fmt=11: both node and edge weights."""
        content = (
            "3 2 11\n"
            "10 2 5\n"
            "20 1 5 3 3\n"
            "30 2 3\n"
        )
        path = _write_tmp(content)
        g = read_metis(path)
        assert g.num_nodes == 3
        assert g.num_edges == 2
        np.testing.assert_array_equal(g.node_weights, [10, 20, 30])
        # Check edge weights for node 1: neighbors 0(w=5), 2(w=3)
        start, end = int(g.xadj[1]), int(g.xadj[2])
        neighbors = g.adjncy[start:end].tolist()
        weights = g.edge_weights[start:end].tolist()
        assert set(zip(neighbors, weights)) == {(0, 5), (2, 3)}

    def test_comments(self):
        """Comments (lines starting with %) should be ignored."""
        content = (
            "% This is a comment\n"
            "% Another comment\n"
            "2 1\n"
            "% inline comment between data\n"
            "2\n"
            "1\n"
        )
        path = _write_tmp(content)
        g = read_metis(path)
        assert g.num_nodes == 2
        assert g.num_edges == 1

    def test_from_metis_classmethod(self):
        """Graph.from_metis should work the same as read_metis."""
        content = (
            "2 1\n"
            "2\n"
            "1\n"
        )
        path = _write_tmp(content)
        g = Graph.from_metis(str(path))
        assert g.num_nodes == 2
        assert g.num_edges == 1


# ------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------

class TestWriteMetis:
    def test_write_unweighted(self, tmp_path):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.finalize()
        out = tmp_path / "test.graph"
        write_metis(g, out)
        text = out.read_text()
        lines = text.strip().split("\n")
        # Header: no fmt since all weights are 1
        assert lines[0] == "3 2"

    def test_write_weighted(self, tmp_path):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1, weight=5)
        g.add_edge(1, 2, weight=3)
        g.set_node_weight(0, 10)
        g.set_node_weight(1, 20)
        g.set_node_weight(2, 30)
        g.finalize()
        out = tmp_path / "test.graph"
        write_metis(g, out)
        text = out.read_text()
        lines = text.strip().split("\n")
        # Header: fmt=11
        assert lines[0] == "3 2 11"

    def test_to_metis_method(self, tmp_path):
        g = Graph(num_nodes=2)
        g.add_edge(0, 1)
        g.finalize()
        out = tmp_path / "test.graph"
        g.to_metis(str(out))
        assert out.exists()


# ------------------------------------------------------------------
# Roundtrip
# ------------------------------------------------------------------

class TestRoundtrip:
    def test_roundtrip_unweighted(self, tmp_path):
        """Write then read back an unweighted graph."""
        g = Graph(num_nodes=4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(0, 3)
        g.finalize()
        out = tmp_path / "test.graph"
        write_metis(g, out)
        g2 = read_metis(out)
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.xadj, g.xadj)
        np.testing.assert_array_equal(g2.adjncy, g.adjncy)
        np.testing.assert_array_equal(g2.node_weights, g.node_weights)
        np.testing.assert_array_equal(g2.edge_weights, g.edge_weights)

    def test_roundtrip_weighted(self, tmp_path):
        """Write then read back a fully weighted graph."""
        g = Graph(num_nodes=4)
        g.add_edge(0, 1, weight=2)
        g.add_edge(1, 2, weight=3)
        g.add_edge(2, 3, weight=4)
        g.set_node_weight(0, 10)
        g.set_node_weight(1, 20)
        g.set_node_weight(2, 30)
        g.set_node_weight(3, 40)
        g.finalize()
        out = tmp_path / "test.graph"
        write_metis(g, out)
        g2 = read_metis(out)
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.xadj, g.xadj)
        np.testing.assert_array_equal(g2.adjncy, g.adjncy)
        np.testing.assert_array_equal(g2.node_weights, g.node_weights)
        np.testing.assert_array_equal(g2.edge_weights, g.edge_weights)

    def test_roundtrip_edge_weights_only(self, tmp_path):
        """Roundtrip with edge weights only (node weights = 1)."""
        g = Graph(num_nodes=3)
        g.add_edge(0, 1, weight=7)
        g.add_edge(1, 2, weight=9)
        g.finalize()
        out = tmp_path / "test.graph"
        write_metis(g, out)
        g2 = read_metis(out)
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.xadj, g.xadj)
        np.testing.assert_array_equal(g2.adjncy, g.adjncy)
        np.testing.assert_array_equal(g2.edge_weights, g.edge_weights)

    def test_roundtrip_node_weights_only(self, tmp_path):
        """Roundtrip with node weights only (edge weights = 1)."""
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.set_node_weight(0, 5)
        g.set_node_weight(1, 10)
        g.set_node_weight(2, 15)
        g.finalize()
        out = tmp_path / "test.graph"
        write_metis(g, out)
        g2 = read_metis(out)
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.xadj, g.xadj)
        np.testing.assert_array_equal(g2.adjncy, g.adjncy)
        np.testing.assert_array_equal(g2.node_weights, g.node_weights)

    def test_roundtrip_conftest_weighted_graph(self, weighted_graph, tmp_path):
        """Roundtrip using the conftest weighted_graph fixture."""
        g = weighted_graph
        out = tmp_path / "test.graph"
        write_metis(g, out)
        g2 = read_metis(out)
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.xadj, g.xadj)
        np.testing.assert_array_equal(g2.adjncy, g.adjncy)
        np.testing.assert_array_equal(g2.edge_weights, g.edge_weights)


# ------------------------------------------------------------------
# Binary I/O
# ------------------------------------------------------------------

class TestGraphBinaryIO:
    """Test Graph.save_binary / Graph.load_binary roundtrips."""

    def test_roundtrip_unweighted(self, tmp_path):
        g = Graph(num_nodes=4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(0, 3)
        g.finalize()
        out = tmp_path / "test.npz"
        g.save_binary(str(out))
        g2 = Graph.load_binary(str(out))
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.xadj, g.xadj)
        np.testing.assert_array_equal(g2.adjncy, g.adjncy)
        np.testing.assert_array_equal(g2.node_weights, g.node_weights)
        np.testing.assert_array_equal(g2.edge_weights, g.edge_weights)

    def test_roundtrip_weighted(self, tmp_path):
        g = Graph(num_nodes=3)
        g.add_edge(0, 1, weight=5)
        g.add_edge(1, 2, weight=3)
        g.set_node_weight(0, 10)
        g.set_node_weight(1, 20)
        g.set_node_weight(2, 30)
        g.finalize()
        out = tmp_path / "test.npz"
        g.save_binary(str(out))
        g2 = Graph.load_binary(str(out))
        assert g2.num_nodes == g.num_nodes
        assert g2.num_edges == g.num_edges
        np.testing.assert_array_equal(g2.node_weights, g.node_weights)
        np.testing.assert_array_equal(g2.edge_weights, g.edge_weights)

    def test_roundtrip_empty(self, tmp_path):
        g = Graph(num_nodes=0)
        g.finalize()
        out = tmp_path / "test.npz"
        g.save_binary(str(out))
        g2 = Graph.load_binary(str(out))
        assert g2.num_nodes == 0
        assert g2.num_edges == 0

    def test_type_mismatch_rejects_hypergraph(self, tmp_path):
        """Loading a hypergraph npz as a Graph should fail."""
        hg = HyperGraph.from_edge_list([[0, 1, 2]], num_nodes=3)
        out = tmp_path / "test.npz"
        hg.save_binary(str(out))
        with pytest.raises(ValueError, match="Expected binary type 1"):
            Graph.load_binary(str(out))


class TestHyperGraphBinaryIO:
    """Test HyperGraph.save_binary / HyperGraph.load_binary roundtrips."""

    def test_roundtrip_unweighted(self, tmp_path):
        hg = HyperGraph.from_edge_list(
            [[0, 1, 2], [1, 2, 3]],
            num_nodes=4,
        )
        out = tmp_path / "test.npz"
        hg.save_binary(str(out))
        hg2 = HyperGraph.load_binary(str(out))
        assert hg2.num_nodes == hg.num_nodes
        assert hg2.num_edges == hg.num_edges
        np.testing.assert_array_equal(hg2.eptr, hg.eptr)
        np.testing.assert_array_equal(hg2.everts, hg.everts)
        np.testing.assert_array_equal(hg2.vptr, hg.vptr)
        np.testing.assert_array_equal(hg2.vedges, hg.vedges)
        np.testing.assert_array_equal(hg2.node_weights, hg.node_weights)
        np.testing.assert_array_equal(hg2.edge_weights, hg.edge_weights)

    def test_roundtrip_weighted(self, tmp_path):
        hg = HyperGraph.from_edge_list(
            [[0, 1], [1, 2, 3]],
            num_nodes=4,
            node_weights=[10, 20, 30, 40],
            edge_weights=[5, 3],
        )
        out = tmp_path / "test.npz"
        hg.save_binary(str(out))
        hg2 = HyperGraph.load_binary(str(out))
        np.testing.assert_array_equal(hg2.node_weights, hg.node_weights)
        np.testing.assert_array_equal(hg2.edge_weights, hg.edge_weights)

    def test_type_mismatch_rejects_graph(self, tmp_path):
        """Loading a graph npz as a HyperGraph should fail."""
        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.finalize()
        out = tmp_path / "test.npz"
        g.save_binary(str(out))
        with pytest.raises(ValueError, match="Expected binary type 2"):
            HyperGraph.load_binary(str(out))


# ------------------------------------------------------------------
# C++ Parser vs Python Parser comparison
# ------------------------------------------------------------------

class TestCppParser:
    """Compare C++ _io parser output against the pure-Python fallback.

    All tests are skipped if the _io C++ extension is not built.
    """

    @pytest.fixture(autouse=True)
    def _require_cpp(self):
        pytest.importorskip("chszlablib._io")

    # -- helpers --

    @staticmethod
    def _assert_graphs_equal(g_cpp, g_py):
        """Assert two Graph objects are structurally identical."""
        assert g_cpp.num_nodes == g_py.num_nodes
        assert g_cpp.num_edges == g_py.num_edges
        np.testing.assert_array_equal(g_cpp.xadj, g_py.xadj)
        np.testing.assert_array_equal(g_cpp.adjncy, g_py.adjncy)
        np.testing.assert_array_equal(g_cpp.node_weights, g_py.node_weights)
        np.testing.assert_array_equal(g_cpp.edge_weights, g_py.edge_weights)

    @staticmethod
    def _assert_hypergraphs_equal(h_cpp, h_py):
        """Assert two HyperGraph objects are structurally identical."""
        assert h_cpp.num_nodes == h_py.num_nodes
        assert h_cpp.num_edges == h_py.num_edges
        np.testing.assert_array_equal(h_cpp.eptr, h_py.eptr)
        np.testing.assert_array_equal(h_cpp.everts, h_py.everts)
        np.testing.assert_array_equal(h_cpp.vptr, h_py.vptr)
        np.testing.assert_array_equal(h_cpp.vedges, h_py.vedges)
        np.testing.assert_array_equal(h_cpp.node_weights, h_py.node_weights)
        np.testing.assert_array_equal(h_cpp.edge_weights, h_py.edge_weights)

    @staticmethod
    def _read_metis_cpp(path):
        from chszlablib._io import read_metis_cpp
        from chszlablib.graph import Graph
        xadj, adjncy, nw, ew = read_metis_cpp(str(path))
        return Graph.from_csr(xadj, adjncy, node_weights=nw, edge_weights=ew)

    @staticmethod
    def _read_hmetis_cpp(path):
        from chszlablib._io import read_hmetis_cpp
        from chszlablib.hypergraph import HyperGraph
        eptr, everts, vptr, vedges, nw, ew, num_nodes = read_hmetis_cpp(str(path))
        return HyperGraph.from_dual_csr(
            vptr=vptr, vedges=vedges,
            eptr=eptr, everts=everts,
            node_weights=nw, edge_weights=ew,
        )

    # -- METIS tests --

    def test_metis_fmt0(self):
        """Unweighted graph (fmt=0 / omitted)."""
        content = "4 4\n2 4\n1 3\n2 4\n1 3\n"
        path = _write_tmp(content)
        self._assert_graphs_equal(
            self._read_metis_cpp(path), _read_metis_python(path)
        )

    def test_metis_fmt1(self):
        """Edge weights only (fmt=1)."""
        content = "3 2 1\n2 5\n1 5 3 3\n2 3\n"
        path = _write_tmp(content)
        self._assert_graphs_equal(
            self._read_metis_cpp(path), _read_metis_python(path)
        )

    def test_metis_fmt10(self):
        """Node weights only (fmt=10)."""
        content = "3 2 10\n10 2\n20 1 3\n30 2\n"
        path = _write_tmp(content)
        self._assert_graphs_equal(
            self._read_metis_cpp(path), _read_metis_python(path)
        )

    def test_metis_fmt11(self):
        """Both node and edge weights (fmt=11)."""
        content = "3 2 11\n10 2 5\n20 1 5 3 3\n30 2 3\n"
        path = _write_tmp(content)
        self._assert_graphs_equal(
            self._read_metis_cpp(path), _read_metis_python(path)
        )

    def test_metis_comments(self):
        """Comments and blank lines should be skipped identically."""
        content = (
            "% comment 1\n"
            "% comment 2\n"
            "2 1\n"
            "% mid-body comment\n"
            "2\n"
            "1\n"
        )
        path = _write_tmp(content)
        self._assert_graphs_equal(
            self._read_metis_cpp(path), _read_metis_python(path)
        )

    def test_metis_isolated_node(self):
        """Graph with an isolated node (empty adjacency line)."""
        content = "3 1\n2\n1\n\n"
        path = _write_tmp(content)
        self._assert_graphs_equal(
            self._read_metis_cpp(path), _read_metis_python(path)
        )

    # -- hMETIS tests --

    def test_hmetis_w0(self):
        """Unweighted hypergraph (W=0 / omitted)."""
        content = "2 4\n1 2 3\n2 3 4\n"
        path = _write_tmp(content)
        self._assert_hypergraphs_equal(
            self._read_hmetis_cpp(path), _read_hmetis_python(path)
        )

    def test_hmetis_w1(self):
        """Edge weights only (W=1)."""
        content = "2 4 1\n5 1 2 3\n3 2 3 4\n"
        path = _write_tmp(content)
        self._assert_hypergraphs_equal(
            self._read_hmetis_cpp(path), _read_hmetis_python(path)
        )

    def test_hmetis_w10(self):
        """Node weights only (W=10)."""
        content = "2 4 10\n1 2 3\n2 3 4\n10\n20\n30\n40\n"
        path = _write_tmp(content)
        self._assert_hypergraphs_equal(
            self._read_hmetis_cpp(path), _read_hmetis_python(path)
        )

    def test_hmetis_w11(self):
        """Both edge and node weights (W=11)."""
        content = "2 4 11\n5 1 2 3\n3 2 3 4\n10\n20\n30\n40\n"
        path = _write_tmp(content)
        self._assert_hypergraphs_equal(
            self._read_hmetis_cpp(path), _read_hmetis_python(path)
        )

    def test_hmetis_comments(self):
        """Comment lines (c and %) should be skipped identically."""
        content = (
            "c a comment\n"
            "% another comment\n"
            "2 3\n"
            "1 2\n"
            "2 3\n"
        )
        path = _write_tmp(content)
        self._assert_hypergraphs_equal(
            self._read_hmetis_cpp(path), _read_hmetis_python(path)
        )

    def test_hmetis_single_vertex_edges(self):
        """Hyperedges containing a single vertex."""
        content = "3 3\n1\n2\n3\n"
        path = _write_tmp(content)
        self._assert_hypergraphs_equal(
            self._read_hmetis_cpp(path), _read_hmetis_python(path)
        )
