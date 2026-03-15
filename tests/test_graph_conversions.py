"""Tests for Graph convenience constructors and conversion methods."""

import numpy as np
import pytest

from chszlablib import Graph


# ------------------------------------------------------------------
# from_edge_list
# ------------------------------------------------------------------

class TestFromEdgeList:
    def test_basic(self):
        g = Graph.from_edge_list([(0, 1), (1, 2), (2, 3)])
        assert g.num_nodes == 4
        assert g.num_edges == 3

    def test_weighted(self):
        g = Graph.from_edge_list([(0, 1, 5), (1, 2, 3)])
        assert g.num_edges == 2
        idx_0 = slice(int(g.xadj[0]), int(g.xadj[1]))
        assert g.edge_weights[idx_0].tolist() == [5]

    def test_explicit_num_nodes(self):
        g = Graph.from_edge_list([(0, 1)], num_nodes=5)
        assert g.num_nodes == 5
        assert g.num_edges == 1

    def test_empty(self):
        g = Graph.from_edge_list([])
        assert g.num_nodes == 0
        assert g.num_edges == 0

    def test_empty_with_nodes(self):
        g = Graph.from_edge_list([], num_nodes=3)
        assert g.num_nodes == 3
        assert g.num_edges == 0

    def test_inferred_num_nodes(self):
        g = Graph.from_edge_list([(0, 5)])
        assert g.num_nodes == 6

    def test_is_finalized(self):
        g = Graph.from_edge_list([(0, 1)])
        with pytest.raises(RuntimeError):
            g.add_edge(0, 1)


# ------------------------------------------------------------------
# from_networkx / to_networkx
# ------------------------------------------------------------------

class TestNetworkX:
    @pytest.fixture(autouse=True)
    def _skip_without_networkx(self):
        pytest.importorskip("networkx")

    def test_round_trip(self):
        import networkx as nx

        G_nx = nx.cycle_graph(5)
        g = Graph.from_networkx(G_nx)
        assert g.num_nodes == 5
        assert g.num_edges == 5

        G_back = g.to_networkx()
        assert G_back.number_of_nodes() == 5
        assert G_back.number_of_edges() == 5

    def test_weighted_round_trip(self):
        import networkx as nx

        G_nx = nx.Graph()
        G_nx.add_edge(0, 1, weight=10)
        G_nx.add_edge(1, 2, weight=20)

        g = Graph.from_networkx(G_nx)
        assert g.num_nodes == 3
        assert g.num_edges == 2

        G_back = g.to_networkx()
        assert G_back[0][1]["weight"] == 10
        assert G_back[1][2]["weight"] == 20

    def test_karate_club(self):
        import networkx as nx

        G_nx = nx.karate_club_graph()
        g = Graph.from_networkx(G_nx)
        assert g.num_nodes == 34
        assert g.num_edges == 78

    def test_directed_raises(self):
        import networkx as nx

        G_nx = nx.DiGraph()
        G_nx.add_edge(0, 1)
        with pytest.raises(TypeError, match="undirected"):
            Graph.from_networkx(G_nx)

    def test_node_weights_preserved(self):
        import networkx as nx

        g = Graph(num_nodes=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.set_node_weight(0, 10)
        g.set_node_weight(2, 20)
        g.finalize()

        G_nx = g.to_networkx()
        assert G_nx.nodes[0]["weight"] == 10
        assert G_nx.nodes[2]["weight"] == 20


# ------------------------------------------------------------------
# from_scipy_sparse / to_scipy_sparse
# ------------------------------------------------------------------

class TestSciPy:
    @pytest.fixture(autouse=True)
    def _skip_without_scipy(self):
        pytest.importorskip("scipy")

    def test_round_trip(self):
        import scipy.sparse as sp

        # 3-node path: 0--1--2
        xadj = np.array([0, 1, 3, 4], dtype=np.int64)
        adjncy = np.array([1, 0, 2, 1], dtype=np.int32)
        data = np.array([1, 1, 1, 1], dtype=np.int64)
        A = sp.csr_array((data, adjncy, xadj), shape=(3, 3))

        g = Graph.from_scipy_sparse(A)
        assert g.num_nodes == 3
        assert g.num_edges == 2

        A_back = g.to_scipy_sparse()
        assert A_back.shape == (3, 3)
        assert A_back.nnz == 4  # both directions

    def test_weighted(self):
        import scipy.sparse as sp

        xadj = np.array([0, 1, 2], dtype=np.int64)
        adjncy = np.array([1, 0], dtype=np.int32)
        data = np.array([5, 5], dtype=np.int64)
        A = sp.csr_array((data, adjncy, xadj), shape=(2, 2))

        g = Graph.from_scipy_sparse(A)
        np.testing.assert_array_equal(g.edge_weights, [5, 5])

    def test_from_csr_matrix(self):
        import scipy.sparse as sp

        A = sp.csr_matrix(np.array([[0, 1], [1, 0]]))
        g = Graph.from_scipy_sparse(A)
        assert g.num_nodes == 2
        assert g.num_edges == 1


# ------------------------------------------------------------------
# Cross-format: edge_list -> scipy -> networkx
# ------------------------------------------------------------------

class TestCrossFormat:
    @pytest.fixture(autouse=True)
    def _skip_without_deps(self):
        pytest.importorskip("networkx")
        pytest.importorskip("scipy")

    def test_edge_list_to_scipy_to_networkx(self):
        import networkx as nx

        g = Graph.from_edge_list([(0, 1), (1, 2), (2, 0)])
        A = g.to_scipy_sparse()
        g2 = Graph.from_scipy_sparse(A)
        G_nx = g2.to_networkx()

        assert G_nx.number_of_nodes() == 3
        assert G_nx.number_of_edges() == 3
