"""Maximum independent set, maximum weight independent set, and b-matching solvers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from chszlablib.graph import Graph

try:
    import gurobipy as _gp  # noqa: F401
    _HYPERMIS_ILP_AVAILABLE = True
    del _gp
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
    is_optimal: bool
    """Whether the solution is provably optimal (requires ILP solver)."""


@dataclass
class BMatchingResult:
    """Result of a hypergraph b-matching computation.

    Given a hypergraph H = (V, E) with edge weights w and vertex capacities b,
    find a set of edges M (matching) that maximizes total weight subject to
    each vertex v being incident to at most b(v) matched edges.
    """

    matched_edges: np.ndarray
    """1-D int array of matched edge indices."""
    total_weight: float
    """Total weight of the matched edges."""
    num_matched: int
    """Number of matched edges."""
    is_optimal: bool
    """Whether the ILP solver proved optimality (only for ``"reductions"``)."""


@dataclass
class TwoPackingResult:
    """Result of a maximum 2-packing set computation.

    Given a graph G = (V, E), a 2-packing set S is a subset of V such
    that no two vertices in S share a common neighbor.  Equivalently, the
    distance between any two vertices in S is at least 3.
    """

    size: int
    """Number of vertices in the 2-packing set."""
    weight: int
    """Total node weight of the selected vertices."""
    vertices: np.ndarray
    """1-D int array of vertex IDs in the 2-packing set."""


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
            "learn_and_reduce": "Maximum weight independent set, GNN-guided kernelization (LearnAndReduce)",
            "two_packing": "Maximum (weighted) 2-packing set (red2pack)",
            "hypermis": "Maximum independent set on hypergraphs (HyperMIS)",
            "bmatching": "Hypergraph b-matching (HeiHGM/Bmatching)",
        }

    HYPERMIS_ILP_AVAILABLE: bool = _HYPERMIS_ILP_AVAILABLE
    """Whether the optional Gurobi ILP solver is available for HyperMIS."""

    LEARN_AND_REDUCE_CONFIGS: tuple[str, ...] = ("cyclic_fast", "cyclic_strong")
    """Valid reduction configuration presets for LearnAndReduce."""

    LEARN_AND_REDUCE_GNN_FILTERS: tuple[str, ...] = (
        "never", "always", "initial", "initial_tight",
    )
    """Valid GNN filter modes for LearnAndReduce."""

    LEARN_AND_REDUCE_SOLVERS: tuple[str, ...] = ("chils", "branch_reduce", "mmwis")
    """Valid kernel solvers for LearnAndReduce full pipeline."""

    BMATCHING_ALGORITHMS: tuple[str, ...] = (
        "greedy_random", "greedy_weight_desc", "greedy_weight_asc",
        "greedy_degree_asc", "greedy_degree_desc",
        "greedy_weight_degree_ratio_desc", "greedy_weight_degree_ratio_asc",
        "reductions", "ils",
    )
    """Valid ``algorithm`` values for :meth:`bmatching`."""

    TWO_PACKING_ALGORITHMS: tuple[str, ...] = (
        "exact", "exact_weighted", "chils", "drp",
        "htwis", "hils", "mmwis", "online", "ilp",
    )
    """Valid ``algorithm`` values for :meth:`two_packing`."""

    # --- KaMIS: Unweighted MIS ---

    @staticmethod
    def redumis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
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

        is_size, is_verts = _redumis(xadj, adjncy, vwgt, time_limit, seed)
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

    # --- LearnAndReduce: GNN-guided MWIS kernelization ---

    @staticmethod
    def learn_and_reduce(
        g: Graph,
        solver: str = "chils",
        config: str = "cyclic_fast",
        gnn_filter: str = "initial_tight",
        time_limit: float = 1000.0,
        solver_time_limit: float = 10.0,
        seed: int = 0,
        num_concurrent: int = 4,
    ) -> MWISResult:
        """Compute MWIS using GNN-guided kernelization (LearnAndReduce).

        Applies learned graph neural network filters to accelerate
        reduction rules, producing a smaller kernel graph. The kernel
        is then solved with the chosen solver and the solution is
        lifted back to the original graph.

        Parameters
        ----------
        g : Graph
            Input graph. Node weights (``g.node_weights``) are required.
        solver : str, optional
            Kernel solver: ``"chils"`` (default), ``"branch_reduce"``,
            or ``"mmwis"``.
        config : str, optional
            Reduction preset: ``"cyclic_fast"`` (default) or
            ``"cyclic_strong"`` (more thorough, slower).
        gnn_filter : str, optional
            GNN filtering mode: ``"initial_tight"`` (default),
            ``"initial"``, ``"always"``, or ``"never"``.
        time_limit : float, optional
            Time limit for kernelization in seconds (default 1000.0).
        solver_time_limit : float, optional
            Time limit for the kernel solver in seconds (default 10.0).
        seed : int, optional
            Random seed (default 0).
        num_concurrent : int, optional
            Number of concurrent threads for CHILS solver (default 4).

        Returns
        -------
        MWISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total weight of the selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *solver*, *config*, or *gnn_filter* is invalid.
        """
        valid_solvers = ("chils", "branch_reduce", "mmwis")
        if solver not in valid_solvers:
            raise ValueError(f"solver must be one of {valid_solvers}, got {solver!r}")

        lr = LearnAndReduceKernel(
            g, config=config, gnn_filter=gnn_filter,
            time_limit=time_limit, seed=seed,
        )
        kernel = lr.kernelize()

        if lr.kernel_nodes == 0:
            return lr.lift_solution(np.array([], dtype=np.int32))

        if solver == "chils":
            kernel_result = IndependenceProblems.chils(
                kernel, time_limit=solver_time_limit,
                num_concurrent=num_concurrent, seed=seed,
            )
        elif solver == "branch_reduce":
            kernel_result = IndependenceProblems.branch_reduce(
                kernel, time_limit=solver_time_limit, seed=seed,
            )
        elif solver == "mmwis":
            kernel_result = IndependenceProblems.mmwis(
                kernel, time_limit=solver_time_limit, seed=seed,
            )

        return lr.lift_solution(kernel_result.vertices)

    # --- HyperMIS: Independent set on hypergraphs ---

    @staticmethod
    def hypermis(
        hg: "HyperGraph",
        time_limit: float = 60.0,
        seed: int = 0,
    ) -> HyperMISResult:
        """Compute a maximum independent set on a hypergraph using HyperMIS.

        Given a hypergraph H = (V, E) where each hyperedge e contains two or
        more vertices, find a maximum independent set I such that for every
        hyperedge e with |e| >= 2, at most one vertex from e is in I.  This is
        "strong" independence: every hyperedge may contribute at most one
        vertex to I.

        Applies strong kernelization reductions (including unconfined vertex
        removal), then solves the remaining kernel exactly via an ILP
        formulation using ``gurobipy``.

        Parameters
        ----------
        hg : HyperGraph
            Input hypergraph.
        time_limit : float, optional
            Wall-clock time budget in seconds (default 60.0).  Also used
            as the Gurobi time limit for the ILP.
        seed : int, optional
            Random seed for reproducibility (default 0).

        Returns
        -------
        HyperMISResult
            ``size`` -- number of vertices in the independent set,
            ``weight`` -- total node weight of selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set,
            ``offset`` -- number of vertices determined during reduction,
            ``reduction_time`` -- wall-clock seconds spent on reductions,
            ``is_optimal`` -- ``True`` if the ILP proved optimality.

        Raises
        ------
        ValueError
            If *time_limit* is negative.
        ImportError
            If ``gurobipy`` is not installed.
        """
        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")
        if not _HYPERMIS_ILP_AVAILABLE:
            raise ImportError(
                "gurobipy is required for HyperMIS. "
                "Install it with: pip install gurobipy"
            )

        hg.finalize()
        eptr = hg.eptr.astype(np.int64, copy=False)
        everts = hg.everts.astype(np.int32, copy=False)

        from chszlablib._hypermis import reduce_and_extract_kernel as _reduce_kernel
        from chszlablib._gurobi_ilp import solve_hypermis_ilp

        (offset, fixed_verts, kernel_eptr, kernel_everts,
         kernel_num_nodes, remap, reduction_time) = _reduce_kernel(
            eptr, everts, hg.num_nodes, time_limit, seed,
            True, False,  # strong_reductions=True, heuristic=False
        )

        # Start with reduction-fixed vertices
        all_verts = list(fixed_verts)

        if kernel_num_nodes > 0:
            remaining_time = max(0.0, time_limit - reduction_time)
            ilp_verts, is_optimal = solve_hypermis_ilp(
                kernel_eptr, kernel_everts, kernel_num_nodes, remaining_time,
            )
            for kv in ilp_verts:
                all_verts.append(int(remap[kv]))
        else:
            is_optimal = True

        is_verts = np.array(sorted(all_verts), dtype=np.int32)
        weight = int(np.sum(hg.node_weights[is_verts])) if hg.node_weights is not None and len(is_verts) > 0 else len(is_verts)
        return HyperMISResult(
            size=len(is_verts),
            weight=weight,
            vertices=is_verts,
            offset=offset,
            reduction_time=reduction_time,
            is_optimal=is_optimal,
        )

    # --- HeiHGM: Hypergraph B-Matching ---

    @staticmethod
    def bmatching(
        hg: "HyperGraph",
        algorithm: str = "greedy_weight_desc",
        seed: int = 0,
        ils_iterations: int = 15,
        ils_time_limit: float = 1800.0,
        ILP_time_limit: float = 1000.0,
    ) -> BMatchingResult:
        """Compute a maximum weight b-matching on a hypergraph.

        Given a hypergraph H = (V, E) with edge weights w and vertex
        capacities b (default 1), find a set of edges M that maximizes
        the total weight sum(w(e) for e in M) subject to each vertex v
        having at most b(v) incident matched edges.

        Parameters
        ----------
        hg : HyperGraph
            Input hypergraph.  Edge weights define the objective.
            Vertex capacities are set via ``hg.set_capacity()`` or
            ``hg.set_capacities()`` (default 1 = standard matching).
        algorithm : str, optional
            Algorithm name (default ``"greedy_weight_desc"``).
            Options: ``"greedy_random"``, ``"greedy_weight_desc"``,
            ``"greedy_weight_asc"``, ``"greedy_degree_asc"``,
            ``"greedy_degree_desc"``, ``"greedy_weight_degree_ratio_desc"``,
            ``"greedy_weight_degree_ratio_asc"``, ``"reductions"``,
            ``"ils"``.
        seed : int, optional
            Random seed (default 0).
        ils_iterations : int, optional
            Max ILS iterations (default 15, only for ``"ils"``).
        ils_time_limit : float, optional
            ILS time budget in seconds (default 1800, only for ``"ils"``).
        ILP_time_limit : float, optional
            ILP time limit in seconds for the ``"reductions"`` algorithm
            (default 1000).

        Returns
        -------
        BMatchingResult
            ``matched_edges`` -- 1-D int array of matched edge indices,
            ``total_weight`` -- total weight of matched edges,
            ``num_matched`` -- number of matched edges.

        Raises
        ------
        ValueError
            If *algorithm* is not recognized.
        """
        from chszlablib.exceptions import InvalidModeError

        if algorithm not in IndependenceProblems.BMATCHING_ALGORITHMS:
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. "
                f"Valid: {IndependenceProblems.BMATCHING_ALGORITHMS}"
            )

        hg.finalize()
        eptr = hg.eptr.astype(np.int64, copy=False)
        everts = hg.everts.astype(np.int32, copy=False)
        edge_weights = hg.edge_weights.astype(np.float64, copy=False)
        capacities = hg.capacities.astype(np.int32, copy=False)

        from chszlablib._bmatching import bmatching as _bmatching

        matched, total_weight, is_optimal = _bmatching(
            eptr, everts, edge_weights, capacities,
            hg.num_nodes, algorithm, seed,
            ils_iterations, ils_time_limit,
            ILP_time_limit,
        )

        return BMatchingResult(
            matched_edges=matched,
            total_weight=total_weight,
            num_matched=len(matched),
            is_optimal=bool(is_optimal),
        )

    # --- red2pack: Maximum 2-Packing Set ---

    @staticmethod
    def two_packing(
        g: Graph,
        algorithm: str = "chils",
        time_limit: float = 100.0,
        seed: int = 0,
        reduction_style: str = "",
    ) -> TwoPackingResult:
        """Compute a maximum (weighted) 2-packing set.

        Given a graph G = (V, E), find a 2-packing set S of maximum weight
        such that no two vertices in S share a common neighbor (equivalently,
        dist(u, v) >= 3 for all u, v in S).

        Uses red2pack's reduce-and-transform strategy: apply 2-packing-set
        reductions, transform to an equivalent MIS problem, and solve with
        the chosen algorithm.

        Parameters
        ----------
        g : Graph
            Input graph. Node weights define the objective; if unset,
            all weights default to 1.
        algorithm : str, optional
            Algorithm name (default ``"chils"``).
            Options: ``"exact"``, ``"exact_weighted"``, ``"chils"``,
            ``"drp"``, ``"htwis"``, ``"hils"``, ``"mmwis"``,
            ``"online"``, ``"ilp"``.
        time_limit : float, optional
            Wall-clock time budget in seconds (default 100.0).
        seed : int, optional
            Random seed (default 0).
        reduction_style : str, optional
            Reduction preset: ``""`` (default from configurator),
            ``"fast"``, ``"strong"``, ``"full"``, ``"heuristic"``.

        Returns
        -------
        TwoPackingResult
            ``size`` -- number of vertices in the 2-packing set,
            ``weight`` -- total weight of selected vertices,
            ``vertices`` -- 1-D int array of vertex IDs in the set.

        Raises
        ------
        ValueError
            If *algorithm* or *time_limit* is invalid.
        """
        from chszlablib.exceptions import InvalidModeError

        if algorithm not in IndependenceProblems.TWO_PACKING_ALGORITHMS:
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. "
                f"Valid: {IndependenceProblems.TWO_PACKING_ALGORITHMS}"
            )
        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = (
            g.node_weights.astype(np.int32, copy=False)
            if g.node_weights is not None
            else np.array([], dtype=np.int32)
        )

        if algorithm == "ilp":
            return IndependenceProblems._two_packing_ilp(
                g, xadj, adjncy, vwgt, time_limit, seed, reduction_style,
            )

        from chszlablib._red2pack import solve_two_packing as _solve

        total_weight, vertices = _solve(
            xadj, adjncy, vwgt, algorithm, time_limit, seed, reduction_style,
        )
        return TwoPackingResult(
            size=len(vertices), weight=int(total_weight), vertices=vertices,
        )

    @staticmethod
    def _two_packing_ilp(
        g: Graph,
        xadj: np.ndarray,
        adjncy: np.ndarray,
        vwgt: np.ndarray,
        time_limit: float,
        seed: int,
        reduction_style: str,
    ) -> TwoPackingResult:
        """Solve 2-packing via kernelization + ILP on the MIS kernel."""
        import gurobipy as gp

        tpk = TwoPackingKernel(
            g, reduction_style=reduction_style,
            time_limit=time_limit, seed=seed,
        )
        kernel = tpk.reduce_and_transform()

        if tpk.kernel_nodes == 0:
            return tpk.lift_solution(np.array([], dtype=np.int32))

        # Build MIS ILP on kernel graph
        kn = kernel.num_nodes
        k_xadj = kernel.xadj
        k_adjncy = kernel.adjncy
        k_wgt = kernel.node_weights if kernel.node_weights is not None else np.ones(kn, dtype=np.int64)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addMVar(kn, vtype=gp.GRB.BINARY, name="x")
                model.setObjective(k_wgt @ x, gp.GRB.MAXIMIZE)

                # Edge constraints: x_u + x_v <= 1
                for u in range(kn):
                    for idx in range(k_xadj[u], k_xadj[u + 1]):
                        v = int(k_adjncy[idx])
                        if u < v:
                            model.addConstr(x[u] + x[v] <= 1)

                model.Params.TimeLimit = max(0.0, time_limit)
                model.optimize()

                mis_verts = np.array(
                    [i for i in range(kn) if x[i].X > 0.5],
                    dtype=np.int32,
                )

        return tpk.lift_solution(mis_verts)


class StreamingBMatcher:
    """True streaming hypergraph matching — process edges one at a time.

    Implements five streaming matching algorithms from HeiHGM/Streaming.
    Each edge is processed on arrival (single pass), making this suitable
    for large-scale hypergraphs that don't fit in memory.

    Parameters
    ----------
    num_nodes : int
        Number of vertices in the hypergraph.
    algorithm : str, optional
        Streaming algorithm (default ``"greedy"``).
    capacities : array-like, optional
        Per-vertex capacities (default all-ones).
    seed : int, optional
        Random seed (default 0).
    epsilon : float, optional
        Approximation parameter for greedy (default 0.0).
    """

    ALGORITHMS: tuple[str, ...] = (
        "naive", "greedy_set", "best_evict", "greedy", "lenient",
    )
    """Valid algorithm names."""

    DEFAULT_ALGORITHM: str = "greedy"
    """Default streaming algorithm (best quality/speed tradeoff)."""

    def __init__(
        self,
        num_nodes: int,
        algorithm: str = "greedy",
        capacities: np.ndarray | list[int] | None = None,
        seed: int = 0,
        epsilon: float = 0.0,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. Valid: {self.ALGORITHMS}"
            )
        self._num_nodes = num_nodes
        self._algorithm = algorithm
        self._seed = seed
        self._epsilon = epsilon
        self._edge_count = 0

        if capacities is not None:
            self._capacities = np.asarray(capacities, dtype=np.int32)
        else:
            self._capacities = np.ones(num_nodes, dtype=np.int32)

        from chszlablib._streaming_bmatching import StreamingMatcher
        self._matcher = StreamingMatcher(
            num_nodes, algorithm, self._capacities, seed, epsilon,
        )

    def add_edge(self, nodes: list[int], weight: float = 1.0) -> None:
        """Feed one hyperedge to the streaming matcher.

        Parameters
        ----------
        nodes : list[int]
            Vertex IDs in this hyperedge.
        weight : float, optional
            Edge weight (default 1.0).
        """
        self._matcher.add_edge(nodes, weight)
        self._edge_count += 1

    def finish(self) -> BMatchingResult:
        """Finalize and return the matching result.

        Returns
        -------
        BMatchingResult
            ``matched_edges`` -- 1-D int array of matched edge indices
            (in insertion order), ``total_weight``, ``num_matched``.
        """
        matched, total_weight = self._matcher.finish()
        return BMatchingResult(
            matched_edges=matched,
            total_weight=total_weight,
            num_matched=len(matched),
            is_optimal=False,
        )

    def reset(self) -> None:
        """Reset internal state for re-streaming."""
        self._matcher.reset()
        self._edge_count = 0

    @property
    def num_edges_streamed(self) -> int:
        """Number of edges fed so far."""
        return self._edge_count

    def __repr__(self) -> str:
        return (
            f"StreamingBMatcher(n={self._num_nodes}, "
            f"algorithm={self._algorithm!r}, "
            f"edges_streamed={self._edge_count})"
        )


class LearnAndReduceKernel:
    """GNN-guided MWIS kernelization with two-step reduce + solve workflow.

    LearnAndReduce uses trained graph neural networks to predict which
    expensive reduction rules will succeed, dramatically speeding up
    preprocessing.  This class exposes the kernelization and solution
    lifting steps separately, letting you solve the reduced kernel with
    any solver.

    Usage::

        lr = LearnAndReduceKernel(g, config="cyclic_fast")
        kernel = lr.kernelize()
        sol = IndependenceProblems.chils(kernel, time_limit=5.0)
        result = lr.lift_solution(sol.vertices)

    Parameters
    ----------
    g : Graph
        Input graph with node weights.
    config : str, optional
        Reduction preset: ``"cyclic_fast"`` (default) or ``"cyclic_strong"``.
    gnn_filter : str, optional
        GNN filtering mode: ``"initial_tight"`` (default), ``"initial"``,
        ``"always"``, or ``"never"``.
    time_limit : float, optional
        Time limit for the kernelization phase in seconds (default 1000.0).
    seed : int, optional
        Random seed (default 0).
    """

    CONFIGS: tuple[str, ...] = ("cyclic_fast", "cyclic_strong")
    GNN_FILTERS: tuple[str, ...] = ("never", "always", "initial", "initial_tight")

    def __init__(
        self,
        g: Graph,
        config: str = "cyclic_fast",
        gnn_filter: str = "initial_tight",
        time_limit: float = 1000.0,
        seed: int = 0,
    ) -> None:
        if config not in self.CONFIGS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"config must be one of {self.CONFIGS}, got {config!r}"
            )
        if gnn_filter not in self.GNN_FILTERS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"gnn_filter must be one of {self.GNN_FILTERS}, got {gnn_filter!r}"
            )

        g.finalize()

        # Resolve models path at runtime
        import importlib.resources
        models_path = str(
            importlib.resources.files("chszlablib").joinpath("models")
        ) + "/"

        from chszlablib._learnandreduce import LearnAndReduceKernel as _LRKernel

        xadj = g.xadj.astype(np.uint32, copy=False)
        adjncy = g.adjncy.astype(np.uint32, copy=False)
        weights = (
            g.node_weights.astype(np.uint64, copy=False)
            if g.node_weights is not None
            else np.ones(g.num_nodes, dtype=np.uint64)
        )

        self._kernel = _LRKernel(
            xadj, adjncy, weights,
            config, gnn_filter, time_limit, seed, models_path,
        )
        self._kernel_graph: Graph | None = None
        self._offset_weight: int = 0
        self._kernel_n: int = -1

    def kernelize(self) -> Graph:
        """Run kernelization and return the reduced kernel as a Graph.

        Returns
        -------
        Graph
            The reduced kernel graph (may have 0 nodes if fully reduced).
        """
        xadj, adjncy, vwgt, offset, kernel_n = self._kernel.kernelize()
        self._offset_weight = int(offset)
        self._kernel_n = int(kernel_n)

        if kernel_n == 0:
            self._kernel_graph = Graph(0)
            self._kernel_graph.finalize()
        else:
            self._kernel_graph = Graph.from_csr(
                xadj, adjncy, node_weights=vwgt,
            )

        return self._kernel_graph

    def lift_solution(self, kernel_vertices: np.ndarray) -> MWISResult:
        """Lift a kernel solution back to the original graph.

        Parameters
        ----------
        kernel_vertices : np.ndarray
            1-D int array of vertex IDs in the kernel's independent set
            (0-indexed in the kernel graph).

        Returns
        -------
        MWISResult
            The full independent set in the original graph.
        """
        kv = np.asarray(kernel_vertices, dtype=np.int32)
        total_weight, vertices = self._kernel.lift_solution(kv)
        return MWISResult(
            size=len(vertices),
            weight=int(total_weight),
            vertices=vertices,
        )

    @property
    def offset_weight(self) -> int:
        """Weight determined by reductions alone (before solving kernel)."""
        return self._offset_weight

    @property
    def kernel_nodes(self) -> int:
        """Number of nodes in the reduced kernel (-1 if not yet kernelized)."""
        return self._kernel_n


class TwoPackingKernel:
    """Two-step 2-packing set solver: reduce-and-transform, then solve MIS kernel.

    Uses red2pack's reduction rules to shrink the 2-packing problem and
    transform it into an equivalent MIS problem on a smaller graph.  You
    solve the kernel MIS with any solver and lift the solution back.

    Usage::

        tpk = TwoPackingKernel(g)
        kernel = tpk.reduce_and_transform()
        sol = IndependenceProblems.branch_reduce(kernel)
        result = tpk.lift_solution(sol.vertices)

    Parameters
    ----------
    g : Graph
        Input graph with optional node weights.
    reduction_style : str, optional
        Reduction preset: ``""`` (default), ``"fast"``, ``"strong"``,
        ``"full"``, ``"heuristic"``.
    time_limit : float, optional
        Time limit for reductions in seconds (default 1000.0).
    seed : int, optional
        Random seed (default 0).
    weighted : bool, optional
        Whether to use weighted reductions (default: auto-detect from
        ``g.node_weights``).
    """

    REDUCTION_STYLES: tuple[str, ...] = ("", "fast", "strong", "full", "heuristic")

    def __init__(
        self,
        g: Graph,
        reduction_style: str = "",
        time_limit: float = 1000.0,
        seed: int = 0,
        weighted: bool | None = None,
    ) -> None:
        g.finalize()

        if weighted is None:
            weighted = g.node_weights is not None

        from chszlablib._red2pack import TwoPackingKernel as _TPKernel

        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = (
            g.node_weights.astype(np.int32, copy=False)
            if g.node_weights is not None
            else np.array([], dtype=np.int32)
        )

        self._kernel = _TPKernel(
            xadj, adjncy, vwgt,
            reduction_style, time_limit, seed, weighted,
        )
        self._kernel_graph: Graph | None = None
        self._offset_weight: int = 0
        self._kernel_n: int = -1

    def reduce_and_transform(self) -> Graph:
        """Run reductions and transform to MIS kernel graph.

        Returns
        -------
        Graph
            The reduced kernel graph (may have 0 nodes if fully reduced).
        """
        solved, xadj, adjncy, vwgt, offset, kernel_n = (
            self._kernel.run_reduce_and_transform()
        )
        self._offset_weight = int(offset)
        self._kernel_n = int(kernel_n)

        if solved or kernel_n == 0:
            self._kernel_graph = Graph(0)
            self._kernel_graph.finalize()
        else:
            self._kernel_graph = Graph.from_csr(
                xadj, adjncy, node_weights=vwgt,
            )

        return self._kernel_graph

    def lift_solution(self, mis_vertices: np.ndarray) -> TwoPackingResult:
        """Lift a kernel MIS solution back to the original 2-packing set.

        Parameters
        ----------
        mis_vertices : np.ndarray
            1-D int array of vertex IDs in the kernel MIS (0-indexed).

        Returns
        -------
        TwoPackingResult
            The full 2-packing set in the original graph.
        """
        kv = np.asarray(mis_vertices, dtype=np.int32)
        total_weight, vertices = self._kernel.lift_solution(kv)
        return TwoPackingResult(
            size=len(vertices),
            weight=int(total_weight),
            vertices=vertices,
        )

    @property
    def offset_weight(self) -> int:
        """Weight determined by reductions alone (before solving kernel)."""
        return self._offset_weight

    @property
    def kernel_nodes(self) -> int:
        """Number of nodes in the kernel (-1 if not yet reduced)."""
        return self._kernel_n
