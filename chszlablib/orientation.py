from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class EdgeOrientationResult:
    """Result of an edge orientation computation."""

    max_out_degree: int
    out_degrees: np.ndarray
    edge_heads: np.ndarray


def orient_edges(
    g: Graph,
    algorithm: str = "combined",
    seed: int = 0,
    eager_size: int = 100,
) -> EdgeOrientationResult:
    """Orient undirected edges to minimize the maximum out-degree.

    Parameters
    ----------
    g : Graph
        Input undirected graph.
    algorithm : str
        Algorithm to use (default ``"combined"``).

        * ``"two_approx"`` -- fast 2-approximation (greedy balanced orientation).
        * ``"dfs"`` -- DFS-based local search improvement.
        * ``"combined"`` -- Eager Path Search (best quality, main contribution).
    seed : int
        Random seed (default 0).
    eager_size : int
        Eager threshold for the combined algorithm (default 100).

    Returns
    -------
    EdgeOrientationResult
        Contains *max_out_degree*, per-node *out_degrees* array, and
        per-CSR-entry *edge_heads* array (1 = oriented away from source
        node, 0 = oriented toward).
    """
    from chszlablib._heiorient import orient_edges as _orient_edges

    g.finalize()

    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    max_out_degree, out_degrees, edge_heads = _orient_edges(
        xadj, adjncy, algorithm, seed, eager_size,
    )

    return EdgeOrientationResult(
        max_out_degree=int(max_out_degree),
        out_degrees=out_degrees,
        edge_heads=edge_heads,
    )
