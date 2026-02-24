"""Edge orientation to minimize maximum out-degree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chszlablib.graph import Graph

OrientationAlgorithm = Literal["two_approx", "dfs", "combined"]

_VALID_ALGORITHMS = {"two_approx", "dfs", "combined"}


@dataclass
class EdgeOrientationResult:
    """Result of an edge orientation computation."""

    max_out_degree: int
    out_degrees: np.ndarray
    edge_heads: np.ndarray


class Orientation:
    """Edge orientation to minimize maximum out-degree."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @staticmethod
    def orient_edges(
        g: Graph,
        algorithm: OrientationAlgorithm = "combined",
        seed: int = 0,
        eager_size: int = 100,
    ) -> EdgeOrientationResult:
        """Orient undirected edges to minimize the maximum out-degree."""
        from chszlablib._heiorient import orient_edges as _orient_edges

        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. "
                f"Choose from: {', '.join(sorted(_VALID_ALGORITHMS))}"
            )

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
