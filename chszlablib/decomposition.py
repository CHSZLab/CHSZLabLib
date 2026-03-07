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
    "inexact": "inexact",
    "exact": "exact",
    "cactus": "cactus",
}

_ORIENTATION_ALGO_MAP = {"two_approx", "dfs", "combined"}

_MOTIF_METHOD_MAP = {"social", "lmchgp"}

_PROCESS_MAP_MODE_MAP = {
    "fast": ("nb_layer", "mtkahypar_default", "kaffpa_fast"),
    "eco": ("nb_layer", "mtkahypar_default", "kaffpa_eco"),
    "strong": ("nb_layer", "mtkahypar_quality", "kaffpa_strong"),
}

_PROCESS_MAP_STRATEGY_SET = {"naive", "layer", "queue", "nb_layer"}

_PROCESS_MAP_ALGORITHM_SET = {
    "kaffpa_fast", "kaffpa_eco", "kaffpa_strong",
    "mtkahypar_default", "mtkahypar_quality", "mtkahypar_highest_quality",
}

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
    "inexact", "exact", "cactus",
]
MaxcutMethod = Literal["heuristic", "exact"]
MotifMethod = Literal["social", "lmchgp"]
HypergraphMincutAlgorithm = Literal["kernelizer", "ilp", "submodular", "trimmer"]
StreamClusterMode = Literal["light", "light_plus", "evo", "strong"]
ProcessMapMode = Literal["fast", "eco", "strong"]

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


@dataclass
class StreamClusterResult:
    """Result of a streaming graph clustering call."""

    modularity: float
    """Estimated modularity score."""

    num_clusters: int
    """Number of clusters found."""

    assignment: np.ndarray
    """Cluster assignment for each node (0-indexed)."""


@dataclass
class HypergraphMincutResult:
    """Result of an exact hypergraph minimum cut computation."""

    cut_value: int
    time: float


@dataclass
class ProcessMappingResult:
    """Result of a hierarchical process mapping computation."""

    comm_cost: int
    """Total communication cost of the mapping."""

    assignment: np.ndarray
    """Process assignment for each node (0-indexed)."""


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
        "inexact", "exact", "cactus",
    )
    """Valid algorithms for :meth:`mincut`."""

    MAXCUT_METHODS: tuple[str, ...] = ("heuristic", "exact")
    """Valid methods for :meth:`maxcut`."""

    HYPERGRAPH_MINCUT_ALGORITHMS: tuple[str, ...] = (
        "kernelizer", "ilp", "submodular", "trimmer",
    )
    """Valid algorithms for :meth:`hypergraph_mincut`."""

    MOTIF_METHODS: tuple[str, ...] = ("social", "lmchgp")
    """Valid methods for :meth:`motif_cluster`."""

    PROCESS_MAP_MODES: tuple[str, ...] = ("fast", "eco", "strong")
    """Valid modes for :meth:`process_map`."""

    PROCESS_MAP_STRATEGIES: tuple[str, ...] = (
        "naive", "layer", "queue", "nb_layer",
    )
    """Valid strategies for :meth:`process_map`."""

    PROCESS_MAP_ALGORITHMS: tuple[str, ...] = (
        "kaffpa_fast", "kaffpa_eco", "kaffpa_strong",
        "mtkahypar_default", "mtkahypar_quality", "mtkahypar_highest_quality",
    )
    """Valid algorithms for :meth:`process_map`."""

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
            "stream_cluster": "Streaming graph clustering (CluStRE)",
            "hypergraph_mincut": "Exact hypergraph minimum cut (HeiCut)",
            "process_map": "Hierarchical process mapping (SharedMap)",
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
        imbalance: float = 0.2,
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

    # --- CluStRE: Streaming Graph Clustering ---

    @staticmethod
    def stream_cluster(
        g: Graph,
        mode: StreamClusterMode = "strong",
        seed: int = 0,
        num_streams_passes: int = 2,
        resolution_param: float = 0.5,
        max_num_clusters: int = -1,
        ls_time_limit: int = 600,
        ls_frac_time: float = 0.5,
        cut_off: float = 0.05,
        suppress_output: bool = True,
    ) -> StreamClusterResult:
        """Cluster a graph using CluStRE's streaming algorithm.

        Uses a streaming model where nodes and their adjacencies are presented
        sequentially, with modularity-based cluster assignment. Supports
        restreaming for improved quality and local search refinement.

        For node-by-node streaming, see :class:`CluStReClusterer`.

        Parameters
        ----------
        g : Graph
            Input graph (undirected, unweighted). Edge weights are ignored.
        mode : str
            Clustering mode. One of ``"light"`` (fastest, single pass),
            ``"light_plus"`` (restreaming + local search), ``"evo"``
            (evolutionary), or ``"strong"`` (best quality, restreaming +
            local search).
        seed : int
            Random seed for reproducibility.
        num_streams_passes : int
            Number of streaming passes (default 2). More passes improve
            quality at the cost of runtime.
        resolution_param : float
            Resolution parameter for modularity (default 0.5). Higher values
            produce more clusters.
        max_num_clusters : int
            Maximum number of clusters. Set to -1 for unlimited.
        ls_time_limit : int
            Local search time limit in seconds (default 600).
        ls_frac_time : float
            Fraction of total time allowed for local search (default 0.5).
        cut_off : float
            Convergence cut-off for local search (default 0.05).
        suppress_output : bool
            If True, suppress stdout/stderr from the C++ algorithm.

        Returns
        -------
        StreamClusterResult
            ``modularity`` (float), ``num_clusters`` (int),
            ``assignment`` (ndarray of cluster IDs, 0-indexed).

        Examples
        --------
        >>> g = Graph.from_edge_list([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)])
        >>> sc = Decomposition.stream_cluster(g, mode="strong")
        >>> print(f"{sc.num_clusters} clusters, modularity={sc.modularity:.4f}")
        """
        from chszlablib._clustre import clustre_cluster

        g.finalize()
        xadj = g.xadj.astype(np.int64)
        adjncy = g.adjncy.astype(np.int64)

        num_clusters, modularity, assignment = clustre_cluster(
            xadj,
            adjncy,
            mode=mode,
            seed=seed,
            num_streams_passes=num_streams_passes,
            resolution_param=resolution_param,
            max_num_clusters=max_num_clusters,
            ls_time_limit=ls_time_limit,
            ls_frac_time=ls_frac_time,
            cut_off=cut_off,
            suppress_output=suppress_output,
        )

        return StreamClusterResult(
            modularity=modularity,
            num_clusters=num_clusters,
            assignment=assignment,
        )

    # --- VieCut: Minimum Cut ---

    @staticmethod
    def mincut(g: Graph, algorithm: MincutAlgorithm = "inexact", seed: int = 0, threads: int = 0) -> MincutResult:
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
            VieCut (heuristic)  ``"inexact"``    Parallel near-linear time; best for large graphs
            Exact               ``"exact"``      Shared-memory parallel exact algorithm
            Cactus              ``"cactus"``     Enumerates all minimum cuts (parallel)
            ==================  ===============  ======================================

        seed : int
            Random seed for reproducibility (used by randomized algorithms).
        threads : int
            Number of threads for parallel algorithms (0 = all available cores).

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
        >>> mc = Decomposition.mincut(g, algorithm="inexact")
        >>> print(f"Min-cut value: {mc.cut_value}")
        """
        from chszlablib._viecut import minimum_cut

        algo_key = algorithm.lower()
        if algo_key not in _MINCUT_ALGO_MAP:
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. "
                f"Choose from: {', '.join(sorted(_MINCUT_ALGO_MAP))}"
            )

        algo_str = _MINCUT_ALGO_MAP[algo_key]

        g.finalize()

        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        adjwgt = g.edge_weights.astype(np.int32, copy=False)

        cut_value, partition = minimum_cut(
            xadj, adjncy, adjwgt,
            algorithm=algo_str,
            save_cut=True,
            seed=seed,
            threads=threads,
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

    # --- HeiCut: Exact Hypergraph Minimum Cut ---

    @staticmethod
    def hypergraph_mincut(
        hg: "HyperGraph",
        algorithm: HypergraphMincutAlgorithm = "kernelizer",
        *,
        base_solver: Literal["submodular", "ilp"] = "submodular",
        ilp_timeout: float = 7200.0,
        ilp_mode: Literal["bip", "milp"] = "bip",
        ordering_type: Literal["ma", "tight", "queyranne"] = "tight",
        ordering_mode: Literal["single", "multi"] = "single",
        seed: int = 0,
        threads: int = 1,
        unweighted: bool = False,
    ) -> HypergraphMincutResult:
        """Compute an exact minimum cut of a hypergraph using HeiCut.

        Finds a bipartition of the vertex set that minimizes the total weight
        of hyperedges crossing the cut. Four algorithms are available:

        ==============  =======================================================
        Algorithm       Description
        ==============  =======================================================
        kernelizer      Kernelization + base solver (fastest in practice)
        ilp             Integer linear programming (requires gurobipy)
        submodular      Submodular function minimization
        trimmer         k-trimmed certificates (unweighted only)
        ==============  =======================================================

        Parameters
        ----------
        hg : HyperGraph
            Input hypergraph (must be finalized).
        algorithm : str
            Algorithm to use. One of ``"kernelizer"``, ``"ilp"``,
            ``"submodular"``, ``"trimmer"``.
        base_solver : str
            Base solver for the kernelizer: ``"submodular"`` (default) or
            ``"ilp"``. Ignored by other algorithms.
        ilp_timeout : float
            Time limit in seconds for ILP-based solving. Applies to
            ``"ilp"`` and ``"kernelizer"`` (with ``base_solver="ilp"``).
        ilp_mode : str
            ILP formulation: ``"bip"`` (binary IP) or ``"milp"`` (mixed ILP).
            Only used by the ``"ilp"`` algorithm.
        ordering_type : str
            Vertex ordering for submodular/trimmer: ``"ma"`` (maximum
            adjacency), ``"tight"``, or ``"queyranne"``.
        ordering_mode : str
            ``"single"`` or ``"multi"`` ordering pass.
        seed : int
            Random seed for reproducibility.
        threads : int
            Number of threads.
        unweighted : bool
            If ``True``, treat all edge weights as 1.

        Returns
        -------
        HypergraphMincutResult
            ``cut_value`` (int) -- minimum edge cut value.
            ``time`` (float) -- computation time in seconds.

        Raises
        ------
        InvalidModeError
            If ``algorithm`` is not recognized.
        ValueError
            If ``threads < 1`` or the hypergraph is empty.
        RuntimeError
            If ILP solving fails (e.g. gurobipy not installed).

        Examples
        --------
        >>> from chszlablib import HyperGraph, Decomposition
        >>> hg = HyperGraph.from_edge_list([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        >>> r = Decomposition.hypergraph_mincut(hg, algorithm="submodular")
        >>> print(f"Min cut: {r.cut_value}")
        """
        from chszlablib.hypergraph import HyperGraph as HG

        if not isinstance(hg, HG):
            raise TypeError(f"Expected HyperGraph, got {type(hg).__name__}")

        valid_algos = {"kernelizer", "ilp", "submodular", "trimmer"}
        if algorithm not in valid_algos:
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. "
                f"Choose from: {', '.join(sorted(valid_algos))}"
            )
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")

        hg.finalize()

        if hg.num_nodes == 0 or hg.num_edges == 0:
            return HypergraphMincutResult(cut_value=0, time=0.0)

        eptr = hg.eptr.astype(np.int64, copy=False)
        everts = hg.everts.astype(np.int32, copy=False)
        nw = hg.node_weights.astype(np.int64, copy=False)
        ew = hg.edge_weights.astype(np.int64, copy=False)

        # Pass empty arrays if all weights are 1 (unit weights)
        if np.all(nw == 1):
            nw = np.empty(0, dtype=np.int64)
        if np.all(ew == 1):
            ew = np.empty(0, dtype=np.int64)

        from chszlablib._heicut import (
            kernelizer as _kernelizer,
            ilp as _ilp,
            submodular as _submodular,
            trimmer as _trimmer,
        )

        if algorithm == "kernelizer":
            cut, t = _kernelizer(
                eptr, everts, nw, ew,
                base_solver=base_solver,
                ilp_timeout=ilp_timeout,
                seed=seed,
                threads=threads,
                unweighted=unweighted,
            )
        elif algorithm == "ilp":
            cut, t = _ilp(
                eptr, everts, nw, ew,
                ilp_timeout=ilp_timeout,
                ilp_mode=ilp_mode,
                seed=seed,
                threads=threads,
                unweighted=unweighted,
            )
        elif algorithm == "submodular":
            cut, t = _submodular(
                eptr, everts, nw, ew,
                ordering_type=ordering_type,
                ordering_mode=ordering_mode,
                seed=seed,
                threads=threads,
                unweighted=unweighted,
            )
        else:  # trimmer
            cut, t = _trimmer(
                eptr, everts, nw, ew,
                ordering_type=ordering_type,
                ordering_mode=ordering_mode,
                seed=seed,
                threads=threads,
            )

        return HypergraphMincutResult(cut_value=int(cut), time=float(t))

    # --- SharedMap: Process Mapping ---

    @staticmethod
    def process_map(
        g: Graph,
        hierarchy: Sequence[int],
        distance: Sequence[int],
        *,
        mode: ProcessMapMode | None = "eco",
        strategy: str | None = None,
        parallel_algorithm: str | None = None,
        serial_algorithm: str | None = None,
        imbalance: float = 0.03,
        threads: int = 1,
        seed: int = 0,
        verbose: bool = False,
    ) -> ProcessMappingResult:
        """Map graph vertices to a hierarchical machine topology using SharedMap.

        Given a communication graph and a hierarchical description of the
        target machine (e.g. 4 nodes x 8 cores), computes an assignment of
        vertices to processing elements that minimizes total communication
        cost. Uses KaHIP for serial partitioning and Mt-KaHyPar for parallel
        partitioning internally.

        Parameters
        ----------
        g : Graph
            Communication graph (undirected, weighted edges represent
            communication volume).
        hierarchy : sequence of int
            Machine hierarchy levels, e.g. ``[4, 8]`` means 4 nodes with 8
            cores each (32 PEs total). The product of all levels determines the
            total number of processing elements.
        distance : sequence of int
            Communication cost per hierarchy level. Must have the same length
            as *hierarchy*. ``distance[i]`` is the cost of communicating across
            level *i* of the hierarchy.
        mode : str or None
            Preset that selects (strategy, parallel_algorithm, serial_algorithm).

            ==========  ==========  ==========  ==========
            Mode        Strategy    Parallel    Serial
            ==========  ==========  ==========  ==========
            ``"fast"``  nb_layer    mtkahypar_default  kaffpa_fast
            ``"eco"``   nb_layer    mtkahypar_default  kaffpa_eco
            ``"strong"``nb_layer    mtkahypar_quality  kaffpa_strong
            ==========  ==========  ==========  ==========

            Set to ``None`` to specify all three individually.
        strategy : str or None
            Thread distribution strategy. Overrides mode preset.
            One of: ``"naive"``, ``"layer"``, ``"queue"``, ``"nb_layer"``.
        parallel_algorithm : str or None
            Parallel partitioning algorithm. Overrides mode preset.
            One of: ``"mtkahypar_default"``, ``"mtkahypar_quality"``,
            ``"mtkahypar_highest_quality"``.
        serial_algorithm : str or None
            Serial partitioning algorithm. Overrides mode preset.
            One of: ``"kaffpa_fast"``, ``"kaffpa_eco"``, ``"kaffpa_strong"``.
        imbalance : float
            Allowed weight imbalance as a fraction (0.03 means 3%).
        threads : int
            Number of threads to use for parallel partitioning.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            If ``True``, print statistics from the C++ algorithm.

        Returns
        -------
        ProcessMappingResult
            ``comm_cost`` (int) -- total communication cost.
            ``assignment`` (ndarray) -- PE assignment for each vertex.

        Raises
        ------
        InvalidModeError
            If ``mode``, ``strategy``, or algorithm strings are not recognized.
        ValueError
            If ``hierarchy`` and ``distance`` have different lengths, or are
            empty, or contain non-positive values.

        Examples
        --------
        >>> g = Graph.from_edge_list([(0,1,10),(1,2,20),(2,3,10),(3,0,20)])
        >>> r = Decomposition.process_map(
        ...     g, hierarchy=[2, 2], distance=[1, 10], mode="fast", threads=4
        ... )
        >>> print(f"Comm cost: {r.comm_cost}, assignment: {r.assignment}")
        """
        from chszlablib._sharedmap import shared_map

        hierarchy = list(hierarchy)
        distance = list(distance)

        if len(hierarchy) != len(distance):
            raise ValueError(
                f"hierarchy and distance must have the same length, "
                f"got {len(hierarchy)} vs {len(distance)}"
            )
        if len(hierarchy) == 0:
            raise ValueError("hierarchy must not be empty")
        if any(h <= 0 for h in hierarchy):
            raise ValueError(f"All hierarchy values must be > 0, got {hierarchy}")
        if any(d < 0 for d in distance):
            raise ValueError(f"All distance values must be >= 0, got {distance}")
        if imbalance < 0:
            raise ValueError(f"imbalance must be >= 0, got {imbalance}")
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")

        # Resolve mode preset
        if mode is not None:
            if mode not in _PROCESS_MAP_MODE_MAP:
                raise InvalidModeError(
                    f"Unknown mode {mode!r}. "
                    f"Choose from: {', '.join(sorted(_PROCESS_MAP_MODE_MAP))}"
                )
            preset_strat, preset_par, preset_ser = _PROCESS_MAP_MODE_MAP[mode]
        else:
            preset_strat = preset_par = preset_ser = None

        strat = strategy if strategy is not None else preset_strat
        par_alg = parallel_algorithm if parallel_algorithm is not None else preset_par
        ser_alg = serial_algorithm if serial_algorithm is not None else preset_ser

        if strat is None or par_alg is None or ser_alg is None:
            raise ValueError(
                "When mode=None, strategy, parallel_algorithm, and "
                "serial_algorithm must all be specified."
            )

        if strat not in _PROCESS_MAP_STRATEGY_SET:
            raise InvalidModeError(
                f"Unknown strategy {strat!r}. "
                f"Choose from: {', '.join(sorted(_PROCESS_MAP_STRATEGY_SET))}"
            )
        if par_alg not in _PROCESS_MAP_ALGORITHM_SET:
            raise InvalidModeError(
                f"Unknown parallel_algorithm {par_alg!r}. "
                f"Choose from: {', '.join(sorted(_PROCESS_MAP_ALGORITHM_SET))}"
            )
        if ser_alg not in _PROCESS_MAP_ALGORITHM_SET:
            raise InvalidModeError(
                f"Unknown serial_algorithm {ser_alg!r}. "
                f"Choose from: {', '.join(sorted(_PROCESS_MAP_ALGORITHM_SET))}"
            )

        g.finalize()
        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        hier_arr = np.array(hierarchy, dtype=np.int32)
        dist_arr = np.array(distance, dtype=np.int32)

        comm_cost, assignment = shared_map(
            vwgt, xadj, adjwgt, adjncy,
            hier_arr, dist_arr,
            float(imbalance), threads, seed,
            strat, par_alg, ser_alg,
            verbose,
        )
        return ProcessMappingResult(comm_cost=comm_cost, assignment=assignment)


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


class CluStReClusterer:
    """Streaming graph clusterer using CluStRE.

    Collects nodes one-by-one via :meth:`new_node`, then clusters the
    accumulated graph when :meth:`cluster` is called.

    Usage::

        cs = CluStReClusterer(mode="strong")
        cs.new_node(0, [1, 2])
        cs.new_node(1, [0, 3])
        cs.new_node(2, [0])
        cs.new_node(3, [1])
        result = cs.cluster()
        print(result.num_clusters, result.modularity)

    Parameters
    ----------
    mode : str
        Clustering mode (``"light"``, ``"light_plus"``, ``"evo"``,
        ``"strong"``).
    seed : int
        Random seed.
    num_streams_passes : int
        Number of streaming passes (default 2).
    resolution_param : float
        Resolution parameter for modularity (default 0.5).
    max_num_clusters : int
        Maximum number of clusters (-1 = unlimited).
    ls_time_limit : int
        Local search time limit in seconds (default 600).
    ls_frac_time : float
        Fraction of total time for local search (default 0.5).
    cut_off : float
        Convergence cut-off for local search (default 0.05).
    suppress_output : bool
        Suppress stdout/stderr from the C++ algorithm.
    """

    def __init__(
        self,
        mode: StreamClusterMode = "strong",
        seed: int = 0,
        num_streams_passes: int = 2,
        resolution_param: float = 0.5,
        max_num_clusters: int = -1,
        ls_time_limit: int = 600,
        ls_frac_time: float = 0.5,
        cut_off: float = 0.05,
        suppress_output: bool = True,
    ):
        self._mode = mode
        self._seed = seed
        self._num_streams_passes = num_streams_passes
        self._resolution_param = resolution_param
        self._max_num_clusters = max_num_clusters
        self._ls_time_limit = ls_time_limit
        self._ls_frac_time = ls_frac_time
        self._cut_off = cut_off
        self._suppress_output = suppress_output

        self._nodes: list[list[int]] = []
        self._node_map: dict[int, int] = {}

    def new_node(self, node: int, neighbors: Sequence[int]) -> None:
        """Add a node with its neighborhood to the stream.

        Nodes can be added in any order and with non-contiguous IDs. Neighbor
        references to nodes not yet added are silently dropped when
        :meth:`cluster` builds the internal CSR representation.

        Parameters
        ----------
        node : int
            The node ID (0-indexed). Must not have been added before.
        neighbors : sequence of int
            Neighbor node IDs (0-indexed).

        Raises
        ------
        ValueError
            If ``node`` has already been added.
        """
        if node in self._node_map:
            raise ValueError(f"Node {node} has already been added")
        self._node_map[node] = len(self._nodes)
        self._nodes.append(list(neighbors))

    def cluster(self) -> StreamClusterResult:
        """Run CluStRE on all added nodes.

        Builds a CSR graph from the nodes added via :meth:`new_node`, runs
        the streaming clusterer, and returns the result.

        Returns
        -------
        StreamClusterResult
            ``modularity``, ``num_clusters``, ``assignment``.
        """
        from chszlablib._clustre import clustre_cluster

        n = len(self._nodes)
        if n == 0:
            return StreamClusterResult(
                modularity=0.0, num_clusters=0,
                assignment=np.array([], dtype=np.int32),
            )

        original_ids = sorted(self._node_map.keys())
        id_to_contiguous = {orig: i for i, orig in enumerate(original_ids)}

        xadj = [0]
        adjncy = []
        for orig_id in original_ids:
            idx = self._node_map[orig_id]
            neighbors = self._nodes[idx]
            mapped = [id_to_contiguous[nb] for nb in neighbors if nb in id_to_contiguous]
            adjncy.extend(mapped)
            xadj.append(len(adjncy))

        xadj_arr = np.array(xadj, dtype=np.int64)
        adjncy_arr = np.array(adjncy, dtype=np.int64)

        num_clusters, modularity, raw_assignment = clustre_cluster(
            xadj_arr,
            adjncy_arr,
            mode=self._mode,
            seed=self._seed,
            num_streams_passes=self._num_streams_passes,
            resolution_param=self._resolution_param,
            max_num_clusters=self._max_num_clusters,
            ls_time_limit=self._ls_time_limit,
            ls_frac_time=self._ls_frac_time,
            cut_off=self._cut_off,
            suppress_output=self._suppress_output,
        )

        if original_ids == list(range(n)):
            assignment = raw_assignment
        else:
            assignment = np.full(max(original_ids) + 1, -1, dtype=np.int32)
            for i, orig_id in enumerate(original_ids):
                assignment[orig_id] = raw_assignment[i]

        return StreamClusterResult(
            modularity=modularity,
            num_clusters=num_clusters,
            assignment=assignment,
        )

    def reset(self) -> None:
        """Clear all added nodes so this clusterer can be reused.

        Retains all configuration parameters but removes every node
        added via :meth:`new_node`.
        """
        self._nodes.clear()
        self._node_map.clear()
