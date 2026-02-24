"""Longest simple path computation via KaLP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class LongestPathResult:
    """Result of a longest path computation."""

    length: int
    path: np.ndarray


def longest_path(
    g: Graph,
    start_vertex: int = 0,
    target_vertex: int = -1,
    partition_config: str = "eco",
    block_size: int = 10,
    number_of_threads: int = 1,
    split_steps: int = 0,
    threshold: int = 0,
) -> LongestPathResult:
    """Compute a longest simple path between two vertices using KaLP.

    Parameters
    ----------
    g : Graph
        Input graph.
    start_vertex : int
        Start vertex (default 0).
    target_vertex : int
        Target vertex (default -1, meaning last vertex n-1).
    partition_config : str
        Partitioning quality: ``"strong"``, ``"eco"`` (default), or ``"fast"``.
    block_size : int
        Block size for the partitioning step (default 10).
    number_of_threads : int
        Number of threads (default 1).
    split_steps : int
        Number of steps to divide the workload (default 0).
    threshold : int
        Boundary-node threshold for parallel block solving (default 0).

    Returns
    -------
    LongestPathResult
        Contains *length* (0 if no path exists) and *path* (vertex sequence).
    """
    from chszlablib._kalp import longest_path as _longest_path

    if partition_config not in ("strong", "eco", "fast"):
        raise ValueError(
            f"Invalid partition_config '{partition_config}'. "
            "Must be 'strong', 'eco', or 'fast'."
        )

    g.finalize()
    n = len(g.xadj) - 1

    if target_vertex < 0:
        target_vertex = n - 1

    if start_vertex < 0 or start_vertex >= n:
        raise ValueError(f"start_vertex {start_vertex} out of range [0, {n - 1}].")
    if target_vertex < 0 or target_vertex >= n:
        raise ValueError(f"target_vertex {target_vertex} out of range [0, {n - 1}].")

    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    ewgt = (
        g.edge_weights.astype(np.int32, copy=False)
        if g.edge_weights is not None
        else np.array([], dtype=np.int32)
    )

    length, path = _longest_path(
        xadj, adjncy, ewgt,
        start_vertex, target_vertex,
        partition_config, block_size,
        number_of_threads, split_steps, threshold,
    )

    return LongestPathResult(length=int(length), path=path)
