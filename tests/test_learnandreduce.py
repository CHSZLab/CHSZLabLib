"""Tests for LearnAndReduce GNN-guided MWIS kernelization."""

import numpy as np
import pytest
from chszlablib import Graph, IndependenceProblems, MWISResult, LearnAndReduceKernel


def make_path_weighted():
    """Weighted path: 0(10)--1(1)--2(10).  Optimal IS = {0,2}, weight=20."""
    g = Graph(3)
    g.set_node_weight(0, 10)
    g.set_node_weight(1, 1)
    g.set_node_weight(2, 10)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def make_star_weighted(n=6, center_weight=1, leaf_weight=10):
    """Star with heavy leaves.  Optimal IS = all leaves."""
    g = Graph(n)
    g.set_node_weight(0, center_weight)
    for i in range(1, n):
        g.set_node_weight(i, leaf_weight)
        g.add_edge(0, i)
    return g


def is_valid_independent_set(g, vertices):
    """Check no two vertices in the set are adjacent."""
    g.finalize()
    vset = set(int(v) for v in vertices)
    for u in vset:
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            if int(g.adjncy[idx]) in vset:
                return False
    return True


# ---- Full pipeline tests ----

class TestLearnAndReduce:
    def test_path(self):
        g = make_path_weighted()
        r = IndependenceProblems.learn_and_reduce(
            g, solver="chils", time_limit=5.0, solver_time_limit=1.0,
        )
        assert isinstance(r, MWISResult)
        assert r.weight >= 20
        assert is_valid_independent_set(g, r.vertices)

    def test_star(self):
        g = make_star_weighted(6)
        r = IndependenceProblems.learn_and_reduce(
            g, solver="chils", time_limit=5.0, solver_time_limit=1.0,
        )
        assert r.weight >= 50
        assert is_valid_independent_set(g, r.vertices)

    def test_solver_branch_reduce(self):
        g = make_path_weighted()
        r = IndependenceProblems.learn_and_reduce(
            g, solver="branch_reduce", time_limit=5.0, solver_time_limit=1.0,
        )
        assert r.weight >= 20
        assert is_valid_independent_set(g, r.vertices)

    def test_solver_mmwis(self):
        g = make_path_weighted()
        r = IndependenceProblems.learn_and_reduce(
            g, solver="mmwis", time_limit=5.0, solver_time_limit=1.0,
        )
        assert r.weight >= 20
        assert is_valid_independent_set(g, r.vertices)

    def test_gnn_never(self):
        g = make_path_weighted()
        r = IndependenceProblems.learn_and_reduce(
            g, gnn_filter="never", time_limit=5.0, solver_time_limit=1.0,
        )
        assert r.weight >= 20

    def test_config_strong(self):
        g = make_path_weighted()
        r = IndependenceProblems.learn_and_reduce(
            g, config="cyclic_strong", time_limit=5.0, solver_time_limit=1.0,
        )
        assert r.weight >= 20

    def test_invalid_solver(self):
        g = make_path_weighted()
        with pytest.raises(ValueError):
            IndependenceProblems.learn_and_reduce(g, solver="invalid")


# ---- Kernelization-only tests ----

class TestLearnAndReduceKernel:
    def test_kernelize(self):
        g = make_path_weighted()
        lr = LearnAndReduceKernel(g, time_limit=5.0)
        kernel = lr.kernelize()
        assert isinstance(kernel, Graph)
        assert lr.offset_weight >= 0
        assert lr.kernel_nodes >= 0

    def test_kernelize_and_lift(self):
        g = make_star_weighted(6)
        lr = LearnAndReduceKernel(g, time_limit=5.0)
        kernel = lr.kernelize()

        if lr.kernel_nodes > 0:
            sol = IndependenceProblems.chils(kernel, time_limit=1.0, num_concurrent=1)
            result = lr.lift_solution(sol.vertices)
        else:
            result = lr.lift_solution(np.array([], dtype=np.int32))

        assert result.weight >= 50
        assert is_valid_independent_set(g, result.vertices)

    def test_fully_reduced(self):
        """Path graph is likely fully reduced by reductions."""
        g = make_path_weighted()
        lr = LearnAndReduceKernel(g, time_limit=5.0)
        kernel = lr.kernelize()

        if lr.kernel_nodes == 0:
            result = lr.lift_solution(np.array([], dtype=np.int32))
            assert result.weight == lr.offset_weight

    def test_invalid_config(self):
        g = make_path_weighted()
        with pytest.raises(Exception):
            LearnAndReduceKernel(g, config="invalid")

    def test_invalid_gnn_filter(self):
        g = make_path_weighted()
        with pytest.raises(Exception):
            LearnAndReduceKernel(g, gnn_filter="invalid")

    def test_larger_graph(self):
        """Cycle graph with 20 nodes."""
        g = Graph(20)
        for i in range(20):
            g.set_node_weight(i, i + 1)
            g.add_edge(i, (i + 1) % 20)
        r = IndependenceProblems.learn_and_reduce(
            g, time_limit=5.0, solver_time_limit=1.0,
        )
        assert r.weight > 0
        assert is_valid_independent_set(g, r.vertices)
