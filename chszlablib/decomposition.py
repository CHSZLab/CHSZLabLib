"""Graph decomposition: partitioning, cuts, clustering, and community detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    partition: np.ndarray | None


@dataclass
class MotifClusterResult:
    """Result of a local motif clustering computation."""

    cluster_nodes: np.ndarray
    motif_conductance: float


# ---------------------------------------------------------------------------
# Decomposition namespace
# ---------------------------------------------------------------------------


class Decomposition:
    """Graph decomposition: partitioning, cuts, clustering, and community detection."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    # --- KaHIP: Graph Partitioning ---

    @staticmethod
    def partition(
        g: Graph,
        num_parts: int = 2,
        mode: str = "eco",
        imbalance: float = 0.03,
        seed: int = 0,
        suppress_output: bool = True,
    ) -> PartitionResult:
        """Partition a graph into *num_parts* blocks using KaHIP."""
        from chszlablib._kahip import kaffpa

        g.finalize()
        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjcwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        edgecut, part = kaffpa(
            vwgt, xadj, adjcwgt, adjncy,
            num_parts, imbalance, suppress_output, seed,
            _MODE_MAP[mode.lower()],
        )
        return PartitionResult(edgecut=edgecut, assignment=part)

    @staticmethod
    def node_separator(
        g: Graph,
        num_parts: int = 2,
        mode: str = "eco",
        imbalance: float = 0.03,
        seed: int = 0,
        suppress_output: bool = True,
    ) -> SeparatorResult:
        """Compute a node separator using KaHIP."""
        from chszlablib._kahip import node_separator as _ns

        g.finalize()
        vwgt = g.node_weights.astype(np.int32, copy=False)
        xadj = g.xadj.astype(np.int32, copy=False)
        adjcwgt = g.edge_weights.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        num_sep, sep = _ns(
            vwgt, xadj, adjcwgt, adjncy,
            num_parts, imbalance, suppress_output, seed,
            _MODE_MAP[mode.lower()],
        )
        return SeparatorResult(num_separator_vertices=num_sep, separator=sep)

    @staticmethod
    def node_ordering(
        g: Graph,
        mode: str = "eco",
        seed: int = 0,
        suppress_output: bool = True,
    ) -> OrderingResult:
        """Compute a reduced nested dissection ordering using KaHIP."""
        from chszlablib._kahip import node_ordering as _no

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)

        ordering = _no(
            xadj, adjncy,
            suppress_output, seed,
            _MODE_MAP[mode.lower()],
        )
        return OrderingResult(ordering=ordering)

    @staticmethod
    def evolutionary_partition(
        g: Graph,
        num_parts: int,
        time_limit: int,
        mode: str = "strong",
        imbalance: float = 0.03,
        seed: int = 0,
        suppress_output: bool = True,
        initial_partition: np.ndarray | None = None,
    ) -> PartitionResult:
        """Partition a graph using KaHIP's evolutionary/memetic algorithm (KaFFPaE)."""
        from chszlablib._kahipe import kaffpaE as _kaffpaE

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
            _KAFFPAE_MODE_MAP[mode.lower()],
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
    ) -> "StreamPartitionResult":
        """Partition a graph using HeiStream's streaming algorithm."""
        from chszlablib._heistream import heistream_partition
        from chszlablib.heistream import StreamPartitionResult

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
    def mincut(g: Graph, algorithm: str = "viecut", seed: int = 0) -> MincutResult:
        """Compute a global minimum cut of an undirected graph."""
        from chszlablib._viecut import minimum_cut

        algo_key = algorithm.lower()
        if algo_key not in _MINCUT_ALGO_MAP:
            raise ValueError(
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
        method: str = "heuristic",
        time_limit: float = 1.0,
    ) -> MaxCutResult:
        """Compute a maximum cut of a graph using FPT kernelization."""
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

        elif method == "exact":
            from chszlablib._maxcut import maxcut_exact

            cut_value, partition = maxcut_exact(
                xadj, adjncy, adjwgt, int(time_limit),
            )
            return MaxCutResult(cut_value=int(cut_value), partition=partition)

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'heuristic' or 'exact'."
            )

    # --- VieClus: Community Detection ---

    @staticmethod
    def cluster(
        g: Graph,
        time_limit: float = 1.0,
        seed: int = 0,
        cluster_upperbound: int = 0,
        suppress_output: bool = True,
    ) -> ClusterResult:
        """Cluster a graph using VieClus (modularity maximization)."""
        from chszlablib._vieclus import cluster as _cluster

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
        """Cluster a signed graph by minimizing disagreements."""
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
        """Cluster a signed graph using memetic evolutionary optimization."""
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
        method: str = "social",
        bfs_depths: list[int] | None = None,
        time_limit: int = 60,
        seed: int = 0,
    ) -> MotifClusterResult:
        """Find a local cluster around a seed node based on triangle motifs."""
        g.finalize()

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
            return MotifClusterResult(
                cluster_nodes=cluster_nodes,
                motif_conductance=float(conductance),
            )

        elif method == "lmchgp":
            from chszlablib._motif import motif_cluster_lmchgp

            cluster_nodes, conductance = motif_cluster_lmchgp(
                xadj, adjncy, seed_node,
                [int(d) for d in bfs_depths], time_limit, seed,
            )
            return MotifClusterResult(
                cluster_nodes=cluster_nodes,
                motif_conductance=float(conductance),
            )

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'social' or 'lmchgp'."
            )
