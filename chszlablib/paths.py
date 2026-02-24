"""Path-based graph problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chszlablib.exceptions import InvalidModeError
from chszlablib.graph import Graph

PartitionConfig = Literal["strong", "eco", "fast"]


@dataclass
class LongestPathResult:
    """Result of a longest path computation."""

    length: int
    path: np.ndarray


class PathProblems:
    """Path-based graph problems."""

    PARTITION_CONFIGS: tuple[str, ...] = ("strong", "eco", "fast")
    """Valid partition configurations for :meth:`longest_path`."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @classmethod
    def available_methods(cls) -> dict[str, str]:
        """Return a dict mapping method names to short descriptions."""
        return {
            "longest_path": "Longest simple path between two vertices (KaLP)",
        }

    @staticmethod
    def longest_path(
        g: Graph,
        start_vertex: int = 0,
        target_vertex: int = -1,
        partition_config: PartitionConfig = "eco",
        block_size: int = 10,
        number_of_threads: int = 1,
        split_steps: int = 0,
        threshold: int = 0,
    ) -> LongestPathResult:
        """Compute a longest simple path between two vertices using KaLP.

        Given an undirected graph G = (V, E) with edge weights w: E -> R>=0
        and two designated vertices s and t, find a simple (vertex-disjoint)
        path P = (s = v0, v1, ..., vl = t) that maximizes the total edge
        weight sum(w({vi, vi+1}) for i = 0..l-1). For unweighted graphs,
        this reduces to finding the path with the most edges.

        The problem is NP-hard. KaLP uses graph partitioning to decompose
        the search space into blocks, then applies dynamic programming
        within and across blocks to find long paths efficiently.

        Parameters
        ----------
        g : Graph
            Input graph. Edge weights (``g.edge_weights``) define the
            objective; if unset, all weights default to 1.
        start_vertex : int, optional
            Source vertex (0-based, default 0).
        target_vertex : int, optional
            Target vertex (0-based, default -1 which maps to the last
            vertex ``n - 1``).
        partition_config : ``"strong"`` | ``"eco"`` | ``"fast"``, optional
            Quality preset for the internal graph partitioning step
            (default ``"eco"``).

            ============ ======== ========= ==========================
            Config       Speed    Quality   Best for
            ============ ======== ========= ==========================
            ``fast``     Fastest  Good      Quick exploration
            ``eco``      Balanced Very good Default choice
            ``strong``   Slowest  Best      Final production results
            ============ ======== ========= ==========================

        block_size : int, optional
            Target block size for partitioning (default 10).
        number_of_threads : int, optional
            Number of threads for parallel search (default 1).
        split_steps : int, optional
            Number of split steps in the search (default 0).
        threshold : int, optional
            Pruning threshold (default 0).

        Returns
        -------
        LongestPathResult
            ``length`` -- total weight (or edge count) of the path found,
            ``path`` -- 1-D int array of vertex IDs along the path.

        Raises
        ------
        InvalidModeError
            If *partition_config* is not one of the valid choices.
        ValueError
            If *start_vertex* or *target_vertex* is out of range.
        """
        from chszlablib._kalp import longest_path as _longest_path

        if partition_config not in ("strong", "eco", "fast"):
            raise InvalidModeError(
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
