"""Tests for METIS I/O (read_metis / write_metis)."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from chszlablib import Graph, read_metis, write_metis


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
