"""Maximum independent set and maximum weight independent set solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph

try:
    from chszlablib._hypermis_ilp import solve as _hypermis_ilp_solve  # type: ignore[import-not-found]
    _HYPERMIS_ILP_AVAILABLE = True
except ImportError:
    _HYPERMIS_ILP_AVAILABLE = False


@dataclass
class MISResult:
    """Result of a maximum independent set computation."""

    size: int
    weight: int
    vertices: np.ndarray


@dataclass
class MWISResult:
    """Result of a maximum weight independent set computation."""

    size: int
    weight: int
    vertices: np.ndarray


@dataclass
class HyperMISResult:
    """Result of a maximum independent set computation on a hypergraph.

    Given a hypergraph H = (V, E) where each hyperedge e contains two or
    more vertices, find a maximum independent set I such that for every
    hyperedge e with |e| >= 2, at most one vertex from e is in I.
    """

    size: int
    """Number of vertices in the independent set."""
    weight: int
    """Total node weight of the selected vertices."""
    vertices: np.ndarray
    """1-D int array of vertex IDs in the independent set."""
    offset: int
    """Number of vertices determined during the reduction phase."""
    reduction_time: float
    """Wall-clock time spent on reductions (seconds)."""


class IndependenceProblems:
    """Maximum independent set and maximum weight independent set solvers."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @classmethod
    def available_methods(cls) -> dict[str, str]:
        """Return a dict mapping method names to short descriptions."""
        return {
            "redumis": "Maximum independent set, evolutionary (KaMIS/ReduMIS)",
            "online_mis": "Maximum independent set, local search (KaMIS/OnlineMIS)",
            "branch_reduce": "Maximum weight independent set, exact (KaMIS/Branch&Reduce)",
            "mmwis": "Maximum weight independent set, evolutionary (KaMIS/MMWIS)",
            "chils": "Maximum weight independent set, concurrent local search (CHILS)",
            "hypermis": "Maximum independent set on hypergraphs, reduction-based (HyperMIS)",
        }

    HYPERMIS_ILP_AVAILABLE: bool = _HYPERMIS_ILP_AVAILABLE
    """Whether the optional Gurobi ILP solver is available for HyperMIS."""

    # --- KaMIS: Unweighted MIS ---

    @staticmethod
    def redumis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
        full_kernelization: bool = False,
    ) -> MISResult:
        """Compute a maximum independent set using the ReduMIS evolutionary algorithm.

        Given an undirected graph G = (V, E), find an independent set I of
        maximum cardinality such that no two vertices in I are adjacent, i.e.,
        maximize |I| subject to {u, v} not in E for all u, v in I.

        The maximum independent set problem is NP-hard and hard to approximate.
        ReduMIS combines graph reduction rules (crown, LP, domination, twin)
        that provably simplify the instance with an evolutionary algorithm that
        operates on the reduced kernel.

        Parameters
        ----------
        g : Graph
            Input graph (unweighted; node weights are ignored for the
            objective but forwarded to the solver).
        time_limit : float, optional
            Wall-clock time budget in seconds (default 10.0).
        seed : int, optional
            Random seed for reproducibility (default 0).
        full_kernelization : bool, optional
            If ``True``, apply the full set of reduction rules before
            starting the evolutionary search (default ``False``).

        Returns
        -------
        MISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total weight of the selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *time_limit* is negative.
        """
        from chszlablib._kamis import redumis as _redumis

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_size, is_verts = _redumis(xadj, adjncy, vwgt, time_limit, seed,
                                     full_kernelization)
        weight = int(np.sum(g.node_weights[is_verts])) if g.node_weights is not None and len(is_verts) > 0 else is_size
        return MISResult(size=is_size, weight=weight, vertices=is_verts)

    @staticmethod
    def online_mis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
        ils_iterations: int = 15000,
    ) -> MISResult:
        """Compute a maximum independent set using OnlineMIS local search.

        Same objective as :meth:`redumis` -- find an independent set of
        maximum cardinality (no two selected vertices are adjacent).
        OnlineMIS uses iterated local search with perturbation and
        incremental updates. It is significantly faster than ReduMIS but
        generally produces smaller independent sets.

        Parameters
        ----------
        g : Graph
            Input graph.
        time_limit : float, optional
            Wall-clock time budget in seconds (default 10.0).
        seed : int, optional
            Random seed for reproducibility (default 0).
        ils_iterations : int, optional
            Number of iterated local search iterations (default 15000).

        Returns
        -------
        MISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total weight of the selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *time_limit* is negative.
        """
        from chszlablib._kamis import online_mis as _online_mis

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_size, is_verts = _online_mis(xadj, adjncy, vwgt, time_limit, seed,
                                        ils_iterations)
        weight = int(np.sum(g.node_weights[is_verts])) if g.node_weights is not None and len(is_verts) > 0 else is_size
        return MISResult(size=is_size, weight=weight, vertices=is_verts)

    # --- KaMIS: Weighted MIS ---

    @staticmethod
    def branch_reduce(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> MISResult:
        """Compute a maximum weight independent set using exact branch-and-reduce.

        Given an undirected graph G = (V, E) with node weights c: V -> R>=0,
        find an independent set I that maximizes the total weight sum(c(v)
        for v in I) subject to {u, v} not in E for all u, v in I.

        Branch & Reduce is an exact solver that applies data reduction rules
        to shrink the instance and then solves the reduced kernel via
        branch-and-bound. It is guaranteed to find an optimal solution but
        may require exponential time in the worst case.

        Parameters
        ----------
        g : Graph
            Input graph. Node weights (``g.node_weights``) define the
            objective; if unset, all weights default to 1 (equivalent to
            the unweighted MIS problem).
        time_limit : float, optional
            Wall-clock time budget in seconds (default 10.0).
        seed : int, optional
            Random seed for reproducibility (default 0).

        Returns
        -------
        MISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total weight of the selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *time_limit* is negative.
        """
        from chszlablib._kamis_wmis import branch_reduce as _branch_reduce

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_weight, is_verts = _branch_reduce(xadj, adjncy, vwgt, time_limit, seed)
        return MISResult(size=len(is_verts), weight=is_weight, vertices=is_verts)

    @staticmethod
    def mmwis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> MISResult:
        """Compute a maximum weight independent set using the MMWIS evolutionary algorithm.

        Same objective as :meth:`branch_reduce` -- find an independent set I
        that maximizes sum(c(v) for v in I) subject to no two vertices in I
        being adjacent. MMWIS uses a memetic evolutionary algorithm: a
        population of independent sets is evolved through recombination and
        local search, guided by reduction rules. Trades exactness for
        scalability on larger instances where branch-and-bound is infeasible.

        Parameters
        ----------
        g : Graph
            Input graph. Node weights (``g.node_weights``) define the
            objective; if unset, all weights default to 1.
        time_limit : float, optional
            Wall-clock time budget in seconds (default 10.0).
        seed : int, optional
            Random seed for reproducibility (default 0).

        Returns
        -------
        MISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total weight of the selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *time_limit* is negative.
        """
        from chszlablib._kamis_mmwis import mmwis_solver as _mmwis_solver

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_weight, is_verts = _mmwis_solver(xadj, adjncy, vwgt, time_limit, seed)
        return MISResult(size=len(is_verts), weight=is_weight, vertices=is_verts)

    # --- CHILS: Weighted MIS ---

    @staticmethod
    def chils(
        g: Graph,
        time_limit: float = 10.0,
        num_concurrent: int = 4,
        seed: int = 0,
    ) -> MWISResult:
        """Compute a maximum weight independent set using CHILS concurrent local search.

        Same objective as :meth:`branch_reduce` -- find an independent set I
        that maximizes sum(c(v) for v in I) subject to no two vertices in I
        being adjacent. CHILS runs multiple concurrent independent local
        searches in parallel, each exploring different regions of the
        solution space. The concurrent design with GNN-accelerated reductions
        makes it particularly effective for large instances where exact
        methods are infeasible.

        Parameters
        ----------
        g : Graph
            Input graph. Node weights (``g.node_weights``) are required and
            define the objective.
        time_limit : float, optional
            Wall-clock time budget in seconds (default 10.0).
        num_concurrent : int, optional
            Number of concurrent local search threads (default 4).
        seed : int, optional
            Random seed for reproducibility (default 0).

        Returns
        -------
        MWISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total weight of the selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *time_limit* is negative or *num_concurrent* is less than 1.
        """
        from chszlablib._chils import mwis as _mwis

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")
        if num_concurrent < 1:
            raise ValueError(f"num_concurrent must be >= 1, got {num_concurrent}")

        g.finalize()
        xadj = g.xadj.astype(np.int64, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        weights = g.node_weights.astype(np.int64, copy=False)

        total_weight, vertices = _mwis(
            xadj, adjncy, weights, time_limit, num_concurrent, seed,
        )
        return MWISResult(size=len(vertices), weight=int(total_weight), vertices=vertices)

    # --- HyperMIS: Independent set on hypergraphs ---

    @staticmethod
    def hypermis(
        hg: "HyperGraph",
        time_limit: float = 60.0,
        seed: int = 0,
        strong_reductions: bool = False,
    ) -> HyperMISResult:
        """Compute a maximum independent set on a hypergraph using HyperMIS reductions.

        Given a hypergraph H = (V, E) where each hyperedge e contains two or
        more vertices, find a maximum independent set I such that for every
        hyperedge e with |e| >= 2, at most one vertex from e is in I.  This is
        "strong" independence: every hyperedge may contribute at most one
        vertex to I.

        HyperMIS applies kernelization reduction rules (vertex domination,
        edge domination, small-edge removal, unconfined vertices) to shrink
        the instance.  Vertices provably in or out of any optimal solution
        are fixed during reduction.

        Parameters
        ----------
        hg : HyperGraph
            Input hypergraph.
        time_limit : float, optional
            Wall-clock time budget for reductions in seconds (default 60.0).
        seed : int, optional
            Random seed for reproducibility (default 0).
        strong_reductions : bool, optional
            If ``True``, enable aggressive reduction rules (unconfined
            vertices, larger edge-size threshold).  Slower but may reduce
            the kernel further (default ``False``).

        Returns
        -------
        HyperMISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total node weight of selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set,
            ``offset`` -- number of vertices determined during reduction,
            ``reduction_time`` -- wall-clock seconds spent on reductions.

        Raises
        ------
        ValueError
            If *time_limit* is negative.
        """
        from chszlablib._hypermis import reduce as _reduce

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        hg.finalize()
        eptr = hg.eptr.astype(np.int64, copy=False)
        everts = hg.everts.astype(np.int32, copy=False)

        offset, is_verts, reduction_time = _reduce(
            eptr, everts, hg.num_nodes, time_limit, seed, strong_reductions,
        )

        weight = int(np.sum(hg.node_weights[is_verts])) if hg.node_weights is not None and len(is_verts) > 0 else len(is_verts)
        return HyperMISResult(
            size=len(is_verts),
            weight=weight,
            vertices=is_verts,
            offset=offset,
            reduction_time=reduction_time,
        )
