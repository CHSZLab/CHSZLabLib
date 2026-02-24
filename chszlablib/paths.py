"""Path-based graph problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class LongestPathResult:
    """Result of a longest path computation."""

    length: int
    path: np.ndarray


class PathProblems:
    """Path-based graph problems."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @staticmethod
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
        """Compute a longest simple path between two vertices using KaLP."""
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
