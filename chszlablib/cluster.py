"""Graph clustering via VieClus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class ClusterResult:
    """Result of a graph clustering computation.

    Attributes
    ----------
    modularity : float
        Modularity of the computed clustering.
    num_clusters : int
        Number of clusters found.
    assignment : ndarray[int32]
        Cluster assignment for each node (length *n*).
    """

    modularity: float
    num_clusters: int
    assignment: np.ndarray


def cluster(
    g: Graph,
    time_limit: float = 1.0,
    seed: int = 0,
    cluster_upperbound: int = 0,
    suppress_output: bool = True,
) -> ClusterResult:
    """Cluster a graph using VieClus.

    Parameters
    ----------
    g : Graph
        The input graph.
    time_limit : float, optional
        Time limit in seconds (default 1.0).
    seed : int, optional
        Random seed (default 0).
    cluster_upperbound : int, optional
        Maximum cluster size; 0 means no limit (default 0).
    suppress_output : bool, optional
        Suppress VieClus console output (default True).

    Returns
    -------
    ClusterResult
        Clustering result with modularity, number of clusters, and
        per-node cluster assignment.
    """
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
