"""Correlation clustering for signed graphs via ScalableCorrelationClustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class CorrelationClusteringResult:
    """Result of a correlation clustering computation."""

    edge_cut: int
    num_clusters: int
    assignment: np.ndarray


def correlation_clustering(
    g: Graph,
    seed: int = 0,
    time_limit: float = 0,
) -> CorrelationClusteringResult:
    """Cluster a signed graph by minimizing disagreements.

    Edges with positive weights represent attraction (should be in same
    cluster) and edges with negative weights represent repulsion (should
    be in different clusters). The algorithm minimizes the total weight
    of violated constraints (the edge cut).

    Parameters
    ----------
    g : Graph
        Input graph with signed edge weights.
    seed : int
        Random seed (default 0).
    time_limit : float
        Time limit in seconds. 0 means single run (default).
        When > 0, repeats and keeps the best solution found.

    Returns
    -------
    CorrelationClusteringResult
        Contains *edge_cut*, *num_clusters*, and *assignment* array.
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


def evolutionary_correlation_clustering(
    g: Graph,
    seed: int = 0,
    time_limit: float = 5.0,
) -> CorrelationClusteringResult:
    """Cluster a signed graph using memetic evolutionary optimization.

    Uses a population-based evolutionary algorithm that repeatedly applies
    multilevel clustering and combines solutions. Generally produces better
    results than single-run ``correlation_clustering`` at the cost of
    longer running time.

    Parameters
    ----------
    g : Graph
        Input graph with signed edge weights.
    seed : int
        Random seed (default 0).
    time_limit : float
        Time limit in seconds (default 5.0).

    Returns
    -------
    CorrelationClusteringResult
        Contains *edge_cut*, *num_clusters*, and *assignment* array.
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
