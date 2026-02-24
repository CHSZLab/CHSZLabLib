"""Graph decomposition: partitioning, cuts, clustering, and community detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from chszlablib.exceptions import InvalidModeError
from chszlablib.graph import Graph

# ---------------------------------------------------------------------------
# Mode maps
# ---------------------------------------------------------------------------

_MODE_MAP = {
    "fast": 0,
    "eco": 1,
    "strong": 2,
    "fastsocial": 3,
    "ecosocial": 4,
    "strongsocial": 5,
}

_KAFFPAE_MODE_MAP = {
    **_MODE_MAP,
    "ultrafastsocial": 6,
}

_MINCUT_ALGO_MAP = {
    "viecut": "vc",
    "vc": "vc",
    "noi": "noi",
    "ks": "ks",
    "matula": "matula",
    "pr": "pr",
    "cactus": "cactus",
}

_ORIENTATION_ALGO_MAP = {"two_approx", "dfs", "combined"}

_MOTIF_METHOD_MAP = {"social", "lmchgp"}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PartitionMode = Literal[
    "fast", "eco", "strong", "fastsocial", "ecosocial", "strongsocial",
]
KaFFPaEMode = Literal[
    "fast", "eco", "strong", "fastsocial", "ecosocial", "strongsocial",
    "ultrafastsocial",
]
MincutAlgorithm = Literal[
    "viecut", "vc", "noi", "ks", "matula", "pr", "cactus",
]
MaxcutMethod = Literal["heuristic", "exact"]
MotifMethod = Literal["social", "lmchgp"]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_mode(mode: str, valid: dict, label: str = "mode") -> int:
    """Validate a mode string and return its integer code."""
    key = mode.lower()
    if key not in valid:
        raise InvalidModeError(
            f"Unknown {label} {mode!r}. "
            f"Choose from: {', '.join(sorted(valid))}"
        )
    return valid[key]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PartitionResult:
    """Result of a graph partitioning call."""

    edgecut: int
    assignment: np.ndarray
    balance: float | None = None


@dataclass
class SeparatorResult:
    """Result of a node separator computation."""

    num_separator_vertices: int
    separator: np.ndarray


@dataclass
class OrderingResult:
    """Result of a nested dissection ordering."""

    ordering: np.ndarray


@dataclass
class MincutResult:
    """Result of a global minimum cut computation."""

    cut_value: int
    partition: np.ndarray


@dataclass
class ClusterResult:
    """Result of a graph clustering computation."""

    modularity: float
    num_clusters: int
    assignment: np.ndarray


@dataclass
class CorrelationClusteringResult:
    """Result of a correlation clustering computation."""

    edge_cut: int
    num_clusters: int
    assignment: np.ndarray


@dataclass
class MaxCutResult:
    """Result of a max-cut computation."""

    cut_value: int
    partition: np.ndarray


@dataclass
class MotifClusterResult:
    """Result of a local motif clustering computation."""

    cluster_nodes: np.ndarray
    motif_conductance: float


@dataclass
class StreamPartitionResult:
    """Result of a streaming graph partitioning call."""

    assignment: np.ndarray
    """Partition assignment for each node (0-indexed)."""


# ---------------------------------------------------------------------------
# Decomposition namespace
# ---------------------------------------------------------------------------


class Decomposition:
    """Graph decomposition: partitioning, cuts, clustering, and community detection."""

    PARTITION_MODES: tuple[str, ...] = (
        "fast", "eco", "strong", "fastsocial", "ecosocial", "strongsocial",
    )
    """Valid modes for :meth:`partition`, :meth:`node_separator`, :meth:`node_ordering`."""

    KAFFPAE_MODES: tuple[str, ...] = (
        "fast", "eco", "strong", "fastsocial", "ecosocial", "strongsocial",
        "ultrafastsocial",
    )
    """Valid modes for :meth:`evolutionary_partition`."""

    MINCUT_ALGORITHMS: tuple[str, ...] = (
        "viecut", "vc", "noi", "ks", "matula", "pr", "cactus",
    )
    """Valid algorithms for :meth:`mincut`."""

    MAXCUT_METHODS: tuple[str, ...] = ("heuristic", "exact")
    """Valid methods for :meth:`maxcut`."""

    MOTIF_METHODS: tuple[str, ...] = ("social", "lmchgp")
    """Valid methods for :meth:`motif_cluster`."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @classmethod
    def available_methods(cls) -> dict[str, str]:
        """Return a dict mapping method names to short descriptions.

        Useful for programmatic discovery of the API.
        """
        return {
            "partition": "Balanced graph partitioning (KaHIP)",
            "evolutionary_partition": "Evolutionary balanced graph partitioning (KaFFPaE)",
            "node_separator": "Balanced node separator (KaHIP)",
            "node_ordering": "Nested dissection ordering (KaHIP)",
            "stream_partition": "Streaming graph partitioning (HeiStream)",
            "mincut": "Global minimum cut (VieCut)",
            "maxcut": "Maximum cut (fpt-max-cut)",
            "cluster": "Community detection / modularity maximization (VieClus)",
            "correlation_clustering": "Correlation clustering on signed graphs (SCC)",
            "evolutionary_correlation_clustering": "Evolutionary correlation clustering (SCC)",
            "motif_cluster": "Local motif clustering (HeidelbergMotifClustering)",
        }

    # --- KaHIP: Graph Partitioning ---

    @staticmethod
    def partition(
        g: Graph,
        num_parts: int = 2,
        mode: PartitionMode = "eco",
        imbalance: float = 0.03,
        seed: int = 0,
        suppress_output: bool = True,
    ) -> PartitionResult:
        """Partition a graph into *num_parts* balanced blocks using KaHIP.

        Finds a partition of the node set into *k* disjoint blocks that minimizes
        the **edge cut** (total weight of edges crossing block boundaries), subject
        to a balance constraint that limits each block's weight to at most
        ``(1 + imbalance) * ceil(total_weight / k)``.

        The problem is NP-hard. KaHIP uses a multilevel approach with local search
        refinement. The ``mode`` parameter controls the quality/speed trade-off.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, optionally weighted).
        num_parts : int
            Number of blocks (must be >= 2).
        mode : str
            Quality/speed trade-off. One of:

            =================  =========  ===========  =============================
            Mode               Speed      Quality      Best for
            =================  =========  ===========  =============================
            ``"fast"``         Fastest    Good         Large-scale exploration
            ``"eco"``          Balanced   Very good    Default choice
            ``"strong"``       Slowest    Best         Final production partitions
            ``"fastsocial"``   Fastest    Good         Social / power-law networks
            ``"ecosocial"``    Balanced   Very good    Social / power-law networks
            ``"strongsocial"`` Slowest    Best         Social / power-law networks
            =================  =========  ===========  =============================

        imbalance : float
            Allowed weight imbalance as a fraction (0.03 means 3%).
        seed : int
            Random seed for reproducibility.
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.

        Returns
        -------
        PartitionResult
            ``edgecut`` (int) -- total weight of edges crossing block boundaries.
            ``assignment`` (ndarray) -- block ID for each node (0-indexed).

        Raises
        ------
        ValueError
            If ``num_parts < 2`` or ``imbalance < 0``.
        InvalidModeError
            If ``mode`` is not recognized.

        Examples
        --------
        >>> g = Graph.from_metis("mesh.graph")
        >>> p = Decomposition.partition(g, num_parts=8, mode="strong", imbalance=0.01)
        >>> print(f"Edgecut: {p.edgecut}")
        """
        from chszlablib._kahip import kaffpa

        if num_parts < 2:
            raise ValueError(f"num_parts must be >= 2, got {num_parts}")
        if imbalance < 0:
            raise ValueError(f"imbalance must be >= 0, got {imbalance}")
        mode_code = _validate_mode(mode, _MODE_MAP)

        g.finalize()
        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjcwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        edgecut, part = kaffpa(
            vwgt, xadj, adjcwgt, adjncy,
            num_parts, imbalance, suppress_output, seed,
            mode_code,
        )
        return PartitionResult(edgecut=edgecut, assignment=part)

    @staticmethod
    def node_separator(
        g: Graph,
        num_parts: int = 2,
        mode: PartitionMode = "eco",
        imbalance: float = 0.03,
        seed: int = 0,
        suppress_output: bool = True,
    ) -> SeparatorResult:
        """Compute a balanced node separator using KaHIP.

        Finds a set S of minimum cardinality such that removing S from the graph
        partitions the remaining nodes into two non-empty sets A and B with no
        edges between them. The balance constraint ensures that
        ``max(|A|, |B|) <= (1 + imbalance) * ceil(|V \\ S| / 2)``.

        Node separators are fundamental in divide-and-conquer algorithms,
        nested dissection orderings for sparse matrix factorization, and VLSI
        design.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, optionally weighted).
        num_parts : int
            Number of blocks (must be >= 2).
        mode : str
            Quality/speed trade-off. Same modes as :meth:`partition`:
            ``"fast"``, ``"eco"``, ``"strong"``, ``"fastsocial"``,
            ``"ecosocial"``, ``"strongsocial"``.
        imbalance : float
            Allowed weight imbalance as a fraction (0.03 means 3%).
        seed : int
            Random seed for reproducibility.
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.

        Returns
        -------
        SeparatorResult
            ``num_separator_vertices`` (int) -- number of nodes in the separator.
            ``separator`` (ndarray) -- array marking each node as belonging to
            block 0, block 1, or the separator (value 2).

        Raises
        ------
        ValueError
            If ``num_parts < 2`` or ``imbalance < 0``.
        InvalidModeError
            If ``mode`` is not recognized.

        Examples
        --------
        >>> g = Graph.from_metis("mesh.graph")
        >>> s = Decomposition.node_separator(g, mode="strong")
        >>> print(f"Separator size: {s.num_separator_vertices}")
        """
        from chszlablib._kahip import node_separator as _ns

        if num_parts < 2:
            raise ValueError(f"num_parts must be >= 2, got {num_parts}")
        if imbalance < 0:
            raise ValueError(f"imbalance must be >= 0, got {imbalance}")
        mode_code = _validate_mode(mode, _MODE_MAP)

        g.finalize()
        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjcwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        num_sep, sep = _ns(
            vwgt, xadj, adjcwgt, adjncy,
            num_parts, imbalance, suppress_output, seed,
            mode_code,
        )
        return SeparatorResult(num_separator_vertices=num_sep, separator=sep)

    @staticmethod
    def node_ordering(
        g: Graph,
        mode: PartitionMode = "eco",
        seed: int = 0,
        suppress_output: bool = True,
    ) -> OrderingResult:
        """Compute a fill-reducing nested dissection ordering using KaHIP.

        Given a sparse symmetric positive-definite matrix (represented as its
        adjacency graph), computes a permutation that minimizes the **fill-in**
        -- the number of new non-zeros introduced during Cholesky factorization.

        The algorithm uses recursive nested dissection: it finds a node
        separator S, orders S last, then recurses on the two disconnected
        subgraphs. High-quality separators (via KaHIP) yield orderings that
        significantly reduce fill-in and factorization time for large sparse
        systems.

        Parameters
        ----------
        g : Graph
            Input graph representing the sparsity pattern of a matrix.
        mode : str
            Quality/speed trade-off. Same modes as :meth:`partition`:
            ``"fast"``, ``"eco"``, ``"strong"``, ``"fastsocial"``,
            ``"ecosocial"``, ``"strongsocial"``.
        seed : int
            Random seed for reproducibility.
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.

        Returns
        -------
        OrderingResult
            ``ordering`` (ndarray) -- permutation array of length ``num_nodes``.
            Node ``i`` should be placed at position ``ordering[i]`` in the
            reordered matrix.

        Raises
        ------
        InvalidModeError
            If ``mode`` is not recognized.

        Examples
        --------
        >>> g = Graph.from_metis("sparse_matrix.graph")
        >>> o = Decomposition.node_ordering(g, mode="strong")
        >>> perm = o.ordering
        """
        from chszlablib._kahip import node_ordering as _no

        mode_code = _validate_mode(mode, _MODE_MAP)

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        ordering = _no(
            xadj, adjncy,
            suppress_output, seed,
            mode_code,
        )
        return OrderingResult(ordering=ordering)

    @staticmethod
    def evolutionary_partition(
        g: Graph,
        num_parts: int,
        time_limit: int,
        mode: KaFFPaEMode = "strong",
        imbalance: float = 0.03,
        seed: int = 0,
        suppress_output: bool = True,
        initial_partition: np.ndarray | None = None,
    ) -> PartitionResult:
        """Partition a graph using KaHIP's evolutionary/memetic algorithm (KaFFPaE).

        Same objective as :meth:`partition` -- minimize the edge cut subject to
        balance constraints -- but solved using a **memetic (evolutionary)
        algorithm**: a population of partitions is maintained and improved
        through recombination operators and multilevel local search over a
        given time budget.

        Supports **warm-starting** from an existing partition via
        ``initial_partition`` to refine a previously computed solution.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, optionally weighted).
        num_parts : int
            Number of blocks (must be >= 2).
        time_limit : int
            Time budget in seconds for the evolutionary search. Longer
            budgets generally yield better partitions.
        mode : str
            Quality/speed trade-off. Same modes as :meth:`partition` plus:

            =====================  =========  ===========
            Mode                   Speed      Quality
            =====================  =========  ===========
            ``"ultrafastsocial"``  Fastest    Baseline
            =====================  =========  ===========

        imbalance : float
            Allowed weight imbalance as a fraction (0.03 means 3%).
        seed : int
            Random seed for reproducibility.
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.
        initial_partition : ndarray or None
            Optional warm-start partition (one block ID per node). If provided,
            the evolutionary algorithm begins from this solution instead of
            computing initial partitions from scratch.

        Returns
        -------
        PartitionResult
            ``edgecut`` (int) -- total weight of edges crossing block boundaries.
            ``assignment`` (ndarray) -- block ID for each node (0-indexed).
            ``balance`` (float) -- achieved balance ratio.

        Raises
        ------
        ValueError
            If ``num_parts < 2``, ``time_limit < 0``, or ``imbalance < 0``.
        InvalidModeError
            If ``mode`` is not recognized.

        Examples
        --------
        >>> g = Graph.from_metis("large_mesh.graph")
        >>> seed_part = Decomposition.partition(g, num_parts=16, mode="eco")
        >>> refined = Decomposition.evolutionary_partition(
        ...     g, num_parts=16, time_limit=60,
        ...     initial_partition=seed_part.assignment,
        ... )
        >>> print(f"Refined edgecut: {refined.edgecut} (balance: {refined.balance:.4f})")
        """
        from chszlablib._kahipe import kaffpaE as _kaffpaE

        if num_parts < 2:
            raise ValueError(f"num_parts must be >= 2, got {num_parts}")
        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")
        if imbalance < 0:
            raise ValueError(f"imbalance must be >= 0, got {imbalance}")
        mode_code = _validate_mode(mode, _KAFFPAE_MODE_MAP)

        g.finalize()
        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjcwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        graph_partitioned = initial_partition is not None
        if initial_partition is not None:
            init_part = np.asarray(initial_partition, dtype=np.int32)
        else:
            init_part = np.empty(0, dtype=np.int32)

        edgecut, balance, part = _kaffpaE(
            vwgt, xadj, adjcwgt, adjncy,
            num_parts, imbalance, suppress_output,
            graph_partitioned, init_part,
            time_limit, seed,
            mode_code,
        )
        return PartitionResult(edgecut=edgecut, assignment=part, balance=balance)

    # --- HeiStream: Streaming Partitioning ---

    @staticmethod
    def stream_partition(
        g: Graph,
        k: int = 2,
        imbalance: float = 3.0,
        seed: int = 0,
        max_buffer_size: int = 0,
        batch_size: int = 0,
        num_streams_passes: int = 1,
        run_parallel: bool = False,
        suppress_output: bool = True,
    ) -> StreamPartitionResult:
        """Partition a graph using HeiStream's streaming algorithm.

        Same objective as :meth:`partition` -- minimize the edge cut subject to
        balance constraints -- but solved in a **streaming** model where nodes
        and their adjacencies are presented sequentially. Each node is assigned
        to a block upon arrival (or after a bounded buffer delay).

        Requires only O(n + B) memory where B is the buffer size, compared
        to O(n + m) for full in-memory partitioning. Supports **Fennel**
        (direct one-pass assignment when ``max_buffer_size`` is 0 or 1),
        **BuffCut** (buffered assignment with local optimization when
        ``max_buffer_size > 1``), and **restreaming** (multiple passes via
        ``num_streams_passes``).

        For node-by-node streaming (e.g., from a network stream or database
        cursor), see :class:`HeiStreamPartitioner`.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, unweighted). Edge weights are ignored.
        k : int
            Number of partitions (must be >= 2).
        imbalance : float
            Allowed imbalance in percent (3.0 means 3%).
        seed : int
            Random seed for reproducibility.
        max_buffer_size : int
            Buffer size for BuffCut. Set to 0 or 1 for direct Fennel mode
            (no buffer). Larger values enable priority-buffer mode.
        batch_size : int
            MLP batch size for model-based partitioning within the buffer.
            Set to 0 for HeiStream's default.
        num_streams_passes : int
            Number of streaming passes (restreaming). More passes improve
            quality at the cost of runtime.
        run_parallel : bool
            Use the parallel 3-thread pipeline (I/O, PQ, partition).
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.

        Returns
        -------
        StreamPartitionResult
            ``assignment`` (ndarray) -- partition ID for each node (0-indexed).

        Raises
        ------
        ValueError
            If ``k < 2`` or ``imbalance < 0``.

        Examples
        --------
        >>> g = Graph.from_metis("large_graph.graph")
        >>> sp = Decomposition.stream_partition(g, k=8, imbalance=3.0)
        >>> print(f"Assignment: {sp.assignment}")
        """
        from chszlablib._heistream import heistream_partition

        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if imbalance < 0:
            raise ValueError(f"imbalance must be >= 0, got {imbalance}")

        g.finalize()
        xadj = g.xadj.astype(np.int64)
        adjncy = g.adjncy.astype(np.int64)

        assignment = heistream_partition(
            xadj,
            adjncy,
            k=k,
            imbalance=imbalance,
            seed=seed,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            num_streams_passes=num_streams_passes,
            run_parallel=run_parallel,
            suppress_output=suppress_output,
        )

        return StreamPartitionResult(assignment=assignment)

    # --- VieCut: Minimum Cut ---

    @staticmethod
    def mincut(g: Graph, algorithm: MincutAlgorithm = "viecut", seed: int = 0) -> MincutResult:
        """Compute a global minimum cut of an undirected graph using VieCut.

        Finds a partition of the node set into two non-empty sets S and V\\S
        that minimizes the total weight of edges crossing the partition. The
        resulting cut value equals the **edge connectivity** of the graph.

        The minimum cut identifies the most vulnerable bottleneck in a network.
        Applications include network reliability analysis, image segmentation,
        and connectivity certification.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, optionally weighted).
        algorithm : str
            Algorithm to use. One of:

            ==================  ===============  ======================================
            Algorithm           Identifier       Characteristics
            ==================  ===============  ======================================
            VieCut              ``"viecut"``     Near-linear time; best for large graphs
            NOI                 ``"noi"``        Deterministic; Nagamochi-Ono-Ibaraki
            Karger-Stein        ``"ks"``         Randomized; Monte Carlo approach
            Matula              ``"matula"``     Approximation-based
            Padberg-Rinaldi     ``"pr"``         Exact; LP-based heuristic
            Cactus              ``"cactus"``     Enumerates all minimum cuts
            ==================  ===============  ======================================

        seed : int
            Random seed for reproducibility (used by randomized algorithms).

        Returns
        -------
        MincutResult
            ``cut_value`` (int) -- weight of the minimum cut.
            ``partition`` (ndarray) -- 0/1 array indicating which side of the
            cut each node belongs to.

        Raises
        ------
        InvalidModeError
            If ``algorithm`` is not recognized.

        Examples
        --------
        >>> g = Graph.from_metis("network.graph")
        >>> mc = Decomposition.mincut(g, algorithm="viecut")
        >>> print(f"Min-cut value: {mc.cut_value}")
        """
        from chszlablib._viecut import minimum_cut

        algo_key = algorithm.lower()
        if algo_key not in _MINCUT_ALGO_MAP:
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. "
                f"Choose from: {', '.join(sorted(_MINCUT_ALGO_MAP))}"
            )

        g.finalize()

        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        adjwgt = g.edge_weights.astype(np.int32, copy=False)

        cut_value, partition = minimum_cut(
            xadj, adjncy, adjwgt,
            algorithm=_MINCUT_ALGO_MAP[algo_key],
            save_cut=True,
            seed=seed,
        )

        return MincutResult(cut_value=int(cut_value), partition=partition)

    # --- MaxCut ---

    @staticmethod
    def maxcut(
        g: Graph,
        method: MaxcutMethod = "heuristic",
        time_limit: float = 1.0,
    ) -> MaxCutResult:
        """Compute a maximum cut of a graph using FPT kernelization.

        Finds a partition of the node set into two sets S and V\\S that
        **maximizes** the total weight of edges crossing the partition. This is
        the dual of the minimum cut problem and is NP-hard.

        The solver applies **FPT kernelization** rules (parameterized by the
        number of edges above the Edwards bound) to reduce the instance,
        followed by either a heuristic or an exact branch-and-bound solver.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, optionally weighted).
        method : str
            Solver method. One of:

            ==============  ===============  ==========================================
            Method          Identifier       Characteristics
            ==============  ===============  ==========================================
            Heuristic       ``"heuristic"``  Fast; good for large graphs
            Exact           ``"exact"``      FPT solver; feasible when kernelization
                                             reduces the instance sufficiently
            ==============  ===============  ==========================================

        time_limit : float
            Time limit in seconds for the solver.

        Returns
        -------
        MaxCutResult
            ``cut_value`` (int) -- weight of the maximum cut.
            ``partition`` (ndarray) -- 0/1 array indicating which side of the
            cut each node belongs to.

        Raises
        ------
        InvalidModeError
            If ``method`` is not ``"heuristic"`` or ``"exact"``.
        ValueError
            If ``time_limit < 0``.

        Examples
        --------
        >>> g = Graph.from_metis("graph.graph")
        >>> mc = Decomposition.maxcut(g, method="heuristic", time_limit=5.0)
        >>> print(f"Max-cut value: {mc.cut_value}")
        """
        if method not in ("heuristic", "exact"):
            raise InvalidModeError(
                f"Unknown method {method!r}. Choose from: heuristic, exact"
            )
        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()

        xadj = g.xadj.astype(np.int64, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        adjwgt = g.edge_weights.astype(np.int64, copy=False)

        if method == "heuristic":
            from chszlablib._maxcut import maxcut_heuristic

            cut_value, partition = maxcut_heuristic(
                xadj, adjncy, adjwgt, time_limit,
            )
            return MaxCutResult(cut_value=int(cut_value), partition=partition)

        else:
            from chszlablib._maxcut import maxcut_exact

            cut_value, partition = maxcut_exact(
                xadj, adjncy, adjwgt, int(time_limit),
            )
            return MaxCutResult(cut_value=int(cut_value), partition=partition)

    # --- VieClus: Community Detection ---

    @staticmethod
    def cluster(
        g: Graph,
        time_limit: float = 1.0,
        seed: int = 0,
        cluster_upperbound: int = 0,
        suppress_output: bool = True,
    ) -> ClusterResult:
        """Cluster a graph using VieClus (modularity maximization).

        Finds a partition of the node set into an automatically determined
        number of clusters that maximizes the **Newman-Girvan modularity**.
        Modularity quantifies the density of edges within clusters relative
        to a random graph with the same degree sequence; values range from
        -0.5 to 1.0 with higher values indicating stronger community structure.

        VieClus uses an evolutionary algorithm with multilevel refinement to
        maximize this objective.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, optionally weighted).
        time_limit : float
            Time budget in seconds for the evolutionary search.
        seed : int
            Random seed for reproducibility.
        cluster_upperbound : int
            Maximum number of clusters allowed (0 means no limit).
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.

        Returns
        -------
        ClusterResult
            ``modularity`` (float) -- achieved modularity value.
            ``num_clusters`` (int) -- number of clusters found.
            ``assignment`` (ndarray) -- cluster ID for each node (0-indexed).

        Raises
        ------
        ValueError
            If ``time_limit < 0``.

        Examples
        --------
        >>> g = Graph.from_metis("social_network.graph")
        >>> c = Decomposition.cluster(g, time_limit=10.0)
        >>> print(f"Modularity: {c.modularity:.4f}, clusters: {c.num_clusters}")
        """
        from chszlablib._vieclus import cluster as _cluster

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()

        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjcwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        modularity, num_clusters, assignment = _cluster(
            vwgt, xadj, adjcwgt, adjncy, suppress_output, seed, time_limit, cluster_upperbound
        )

        return ClusterResult(
            modularity=modularity,
            num_clusters=num_clusters,
            assignment=assignment,
        )

    # --- SCC: Correlation Clustering ---

    @staticmethod
    def correlation_clustering(
        g: Graph,
        seed: int = 0,
        time_limit: float = 0,
    ) -> CorrelationClusteringResult:
        """Cluster a signed graph by minimizing disagreements using SCC.

        Given a graph with **signed** edge weights (positive = similarity,
        negative = dissimilarity), finds a partition into an arbitrary number
        of clusters that minimizes the total **disagreements**: positive edges
        crossing cluster boundaries plus negative edges within clusters.

        Unlike standard clustering, the number of clusters is not fixed but
        determined automatically by the optimization. SCC uses multilevel label
        propagation to solve this efficiently.

        Parameters
        ----------
        g : Graph
            Input graph with signed edge weights. Positive weights indicate
            similarity, negative weights indicate dissimilarity.
        seed : int
            Random seed for reproducibility.
        time_limit : float
            Time limit in seconds (0 means no limit).

        Returns
        -------
        CorrelationClusteringResult
            ``edge_cut`` (int) -- number of disagreements.
            ``num_clusters`` (int) -- number of clusters found.
            ``assignment`` (ndarray) -- cluster ID for each node (0-indexed).

        Examples
        --------
        >>> g = Graph(num_nodes=4)
        >>> g.add_edge(0, 1, weight=1)   # similar
        >>> g.add_edge(1, 2, weight=-1)  # dissimilar
        >>> g.add_edge(2, 3, weight=1)   # similar
        >>> r = Decomposition.correlation_clustering(g, seed=42)
        >>> print(f"Clusters: {r.num_clusters}, disagreements: {r.edge_cut}")
        """
        from chszlablib._scc import correlation_clustering as _cc

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        adjwgt = g.edge_weights.astype(np.int32, copy=False) if g.edge_weights is not None else np.array([], dtype=np.int32)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        edge_cut, num_clusters, assignment = _cc(xadj, adjncy, adjwgt, vwgt,
                                                 seed, time_limit)
        return CorrelationClusteringResult(
            edge_cut=edge_cut,
            num_clusters=num_clusters,
            assignment=assignment,
        )

    @staticmethod
    def evolutionary_correlation_clustering(
        g: Graph,
        seed: int = 0,
        time_limit: float = 5.0,
    ) -> CorrelationClusteringResult:
        """Cluster a signed graph using memetic evolutionary optimization (SCC).

        Same objective as :meth:`correlation_clustering` -- minimize
        disagreements on a signed graph -- but solved using a
        **population-based memetic evolutionary algorithm** that maintains a
        pool of clusterings and improves them through recombination and
        multilevel local search over a given time budget. Yields
        higher-quality solutions at the cost of increased runtime.

        Parameters
        ----------
        g : Graph
            Input graph with signed edge weights. Positive weights indicate
            similarity, negative weights indicate dissimilarity.
        seed : int
            Random seed for reproducibility.
        time_limit : float
            Time budget in seconds for the evolutionary search.

        Returns
        -------
        CorrelationClusteringResult
            ``edge_cut`` (int) -- number of disagreements.
            ``num_clusters`` (int) -- number of clusters found.
            ``assignment`` (ndarray) -- cluster ID for each node (0-indexed).

        Examples
        --------
        >>> g = Graph.from_metis("signed_graph.graph")
        >>> r = Decomposition.evolutionary_correlation_clustering(g, time_limit=30.0)
        >>> print(f"Clusters: {r.num_clusters}, disagreements: {r.edge_cut}")
        """
        from chszlablib._scc_evo import evolutionary_correlation_clustering as _ecc

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        adjwgt = g.edge_weights.astype(np.int32, copy=False) if g.edge_weights is not None else np.array([], dtype=np.int32)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        edge_cut, num_clusters, assignment = _ecc(xadj, adjncy, adjwgt, vwgt,
                                                  seed, time_limit)
        return CorrelationClusteringResult(
            edge_cut=edge_cut,
            num_clusters=num_clusters,
            assignment=assignment,
        )

    # --- Motif Clustering ---

    @staticmethod
    def motif_cluster(
        g: Graph,
        seed_node: int,
        method: MotifMethod = "social",
        bfs_depths: list[int] | None = None,
        time_limit: int = 60,
        seed: int = 0,
    ) -> MotifClusterResult:
        """Find a local cluster around a seed node based on triangle motifs.

        Given a seed node, finds a cluster containing that node which minimizes
        the **triangle-motif conductance**: the ratio of triangles cut by the
        cluster boundary to the minimum of triangles inside vs. outside the
        cluster.

        Unlike global clustering, this operates **locally** -- the algorithm
        explores only the neighborhood of the seed node via BFS and does not
        need to process the entire graph. Applications include community
        detection around a query node in social networks.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, unweighted).
        seed_node : int
            The node around which to find a local cluster (0-indexed).
        method : str
            Clustering method. One of:

            ==========  =============  ===================================
            Method      Identifier     Characteristics
            ==========  =============  ===================================
            SOCIAL      ``"social"``   Flow-based; faster
            LMCHGP      ``"lmchgp"``   Graph-partitioning-based
            ==========  =============  ===================================

        bfs_depths : list of int or None
            BFS depths to explore around the seed node. Controls the size of
            the local subgraph considered. Defaults to ``[10, 15, 20]``.
        time_limit : int
            Time limit in seconds for the computation.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        MotifClusterResult
            ``cluster_nodes`` (ndarray) -- node IDs in the found cluster.
            ``motif_conductance`` (float) -- triangle-motif conductance of the
            cluster (lower is better).

        Raises
        ------
        InvalidModeError
            If ``method`` is not recognized.
        ValueError
            If ``seed_node`` is out of range or ``time_limit < 0``.

        Examples
        --------
        >>> g = Graph.from_metis("social_network.graph")
        >>> r = Decomposition.motif_cluster(g, seed_node=42, method="social")
        >>> print(f"Cluster size: {len(r.cluster_nodes)}, conductance: {r.motif_conductance:.4f}")
        """
        if method not in _MOTIF_METHOD_MAP:
            raise InvalidModeError(
                f"Unknown method {method!r}. "
                f"Choose from: {', '.join(sorted(_MOTIF_METHOD_MAP))}"
            )
        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()

        if seed_node < 0 or seed_node >= g.num_nodes:
            raise ValueError(
                f"seed_node {seed_node} out of range [0, {g.num_nodes})"
            )

        if bfs_depths is None:
            bfs_depths = [10, 15, 20]

        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        if method == "social":
            from chszlablib._motif import motif_cluster_social

            cluster_nodes, conductance = motif_cluster_social(
                xadj, adjncy, seed_node,
                [int(d) for d in bfs_depths], time_limit, seed,
            )
        else:
            from chszlablib._motif import motif_cluster_lmchgp

            cluster_nodes, conductance = motif_cluster_lmchgp(
                xadj, adjncy, seed_node,
                [int(d) for d in bfs_depths], time_limit, seed,
            )

        return MotifClusterResult(
            cluster_nodes=cluster_nodes,
            motif_conductance=float(conductance),
        )


# ---------------------------------------------------------------------------
# HeiStreamPartitioner (stateful streaming class)
# ---------------------------------------------------------------------------


class HeiStreamPartitioner:
    """Streaming graph partitioner using HeiStream.

    Supports both the full BuffCut algorithm (with configurable buffer and
    window/batch sizes) and the simpler direct Fennel one-pass mode.

    Usage::

        hs = HeiStreamPartitioner(k=4, imbalance=3.0, max_buffer_size=1000)
        hs.new_node(0, [1, 2])
        hs.new_node(1, [0, 3])
        hs.new_node(2, [0])
        hs.new_node(3, [1])
        result = hs.partition()
        print(result.assignment)   # array of partition IDs

    Parameters
    ----------
    k : int
        Number of partitions.
    imbalance : float
        Allowed imbalance in percent (e.g. 3.0 means 3%).
    seed : int
        Random seed.
    max_buffer_size : int
        Buffer size for BuffCut. Set to 0 or 1 for direct Fennel (no buffer).
        Larger values enable the priority-buffer mode.
    batch_size : int
        MLP batch size for model-based partitioning within the buffer.
        Set to 0 for HeiStream's default.
    num_streams_passes : int
        Number of streaming passes (restreaming).
    run_parallel : bool
        Use the parallel 3-thread pipeline (I/O, PQ, partition).
    suppress_output : bool
        Suppress stdout/stderr from the C++ algorithm.
    """

    def __init__(
        self,
        k: int = 2,
        imbalance: float = 3.0,
        seed: int = 0,
        max_buffer_size: int = 0,
        batch_size: int = 0,
        num_streams_passes: int = 1,
        run_parallel: bool = False,
        suppress_output: bool = True,
    ):
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if imbalance < 0:
            raise ValueError(f"imbalance must be >= 0, got {imbalance}")

        self._k = k
        self._imbalance = imbalance
        self._seed = seed
        self._max_buffer_size = max_buffer_size
        self._batch_size = batch_size
        self._num_streams_passes = num_streams_passes
        self._run_parallel = run_parallel
        self._suppress_output = suppress_output

        self._nodes: list[list[int]] = []
        self._node_map: dict[int, int] = {}

    def new_node(self, node: int, neighbors: Sequence[int]) -> None:
        """Add a node with its neighborhood to the stream.

        Nodes can be added in any order and with non-contiguous IDs. Neighbor
        references to nodes not yet added (or never added) are silently
        dropped when :meth:`partition` builds the internal CSR representation.

        Parameters
        ----------
        node : int
            The node ID (0-indexed). Must not have been added before.
        neighbors : sequence of int
            Neighbor node IDs (0-indexed). Edges are undirected; adding
            ``new_node(0, [1])`` and ``new_node(1, [0])`` creates one edge.

        Raises
        ------
        ValueError
            If ``node`` has already been added.
        """
        if node in self._node_map:
            raise ValueError(f"Node {node} has already been added")
        self._node_map[node] = len(self._nodes)
        self._nodes.append(list(neighbors))

    def partition(self) -> StreamPartitionResult:
        """Run the HeiStream algorithm on all added nodes.

        Builds a CSR graph from the nodes added via :meth:`new_node`, remaps
        non-contiguous node IDs to a contiguous range internally, runs the
        streaming partitioner, and maps the result back to the original IDs.

        If no nodes have been added, returns an empty assignment array.

        Returns
        -------
        StreamPartitionResult
            ``assignment`` (ndarray) -- partition ID for each node (0-indexed).
            For non-contiguous node IDs, the array length equals
            ``max(node_id) + 1`` and unregistered positions are set to -1.

        Examples
        --------
        >>> hs = HeiStreamPartitioner(k=2, imbalance=3.0)
        >>> hs.new_node(0, [1, 2])
        >>> hs.new_node(1, [0])
        >>> hs.new_node(2, [0])
        >>> result = hs.partition()
        >>> print(result.assignment)
        """
        from chszlablib._heistream import heistream_partition

        n = len(self._nodes)
        if n == 0:
            return StreamPartitionResult(assignment=np.array([], dtype=np.int32))

        # Build CSR representation
        # First remap node IDs to contiguous 0..n-1 if needed
        original_ids = sorted(self._node_map.keys())
        id_to_contiguous = {orig: i for i, orig in enumerate(original_ids)}

        xadj = [0]
        adjncy = []
        for orig_id in original_ids:
            idx = self._node_map[orig_id]
            neighbors = self._nodes[idx]
            mapped = []
            for nb in neighbors:
                if nb in id_to_contiguous:
                    mapped.append(id_to_contiguous[nb])
            adjncy.extend(mapped)
            xadj.append(len(adjncy))

        xadj_arr = np.array(xadj, dtype=np.int64)
        adjncy_arr = np.array(adjncy, dtype=np.int64)

        raw_assignment = heistream_partition(
            xadj_arr,
            adjncy_arr,
            k=self._k,
            imbalance=self._imbalance,
            seed=self._seed,
            max_buffer_size=self._max_buffer_size,
            batch_size=self._batch_size,
            num_streams_passes=self._num_streams_passes,
            run_parallel=self._run_parallel,
            suppress_output=self._suppress_output,
        )

        # Map back to original node IDs if non-contiguous
        if original_ids == list(range(n)):
            assignment = raw_assignment
        else:
            assignment = np.full(max(original_ids) + 1, -1, dtype=np.int32)
            for i, orig_id in enumerate(original_ids):
                assignment[orig_id] = raw_assignment[i]

        return StreamPartitionResult(assignment=assignment)

    def reset(self) -> None:
        """Clear all added nodes so this partitioner can be reused.

        Retains all configuration parameters (k, imbalance, seed, etc.)
        but removes every node added via :meth:`new_node`. Call this to
        partition a different graph without constructing a new instance.
        """
        self._nodes.clear()
        self._node_map.clear()
