from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class MotifClusterResult:
    """Result of a local motif clustering computation."""

    cluster_nodes: np.ndarray
    motif_conductance: float


def motif_cluster(
    g: Graph,
    seed_node: int,
    method: str = "social",
    bfs_depths: list[int] | None = None,
    time_limit: int = 60,
    seed: int = 0,
) -> MotifClusterResult:
    """Find a local cluster around a seed node based on triangle motifs.

    Parameters
    ----------
    g : Graph
        Input undirected, unweighted graph.
    seed_node : int
        Node around which to find the cluster (0-indexed).
    method : str
        ``"social"`` (default) -- faster flow-based approach (ESA 2023).
        ``"lmchgp"`` -- graph-partitioning-based approach (ALENEX 2023).
    bfs_depths : list[int], optional
        BFS neighborhood depths to try (default ``[10, 15, 20]``).
    time_limit : int
        Time limit in seconds (default 60).
    seed : int
        Random seed (default 0).

    Returns
    -------
    MotifClusterResult
        *cluster_nodes* is an array of node IDs in the cluster.
        *motif_conductance* is the motif conductance score (lower is better).
    """
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
