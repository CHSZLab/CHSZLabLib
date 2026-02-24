"""Edge orientation to minimize maximum out-degree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chszlablib.exceptions import InvalidModeError
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

    ALGORITHMS: tuple[str, ...] = ("two_approx", "dfs", "combined")
    """Valid algorithms for :meth:`orient_edges`."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @classmethod
    def available_methods(cls) -> dict[str, str]:
        """Return a dict mapping method names to short descriptions."""
        return {
            "orient_edges": "Edge orientation to minimize max out-degree (HeiOrient)",
        }

    @staticmethod
    def orient_edges(
        g: Graph,
        algorithm: OrientationAlgorithm = "combined",
        seed: int = 0,
        eager_size: int = 100,
    ) -> EdgeOrientationResult:
        """Orient undirected edges to minimize the maximum out-degree.

        Given an undirected graph G = (V, E), orient each edge (assign a
        direction) to obtain a directed graph such that the maximum
        out-degree over all vertices is minimized. The optimal value equals
        the arboricity of the graph, defined as
        max over all subgraphs H of ceil(|E(H)| / (|V(H)| - 1)).

        Low out-degree orientations enable space-efficient data structures
        for adjacency queries, fast triangle enumeration, and compact graph
        representations.

        Parameters
        ----------
        g : Graph
            Input undirected graph.
        algorithm : ``"two_approx"`` | ``"dfs"`` | ``"combined"``, optional
            Algorithm to use (default ``"combined"``).

            ============== ===========================================
            Algorithm      Characteristics
            ============== ===========================================
            ``two_approx`` Fast greedy; guaranteed 2-approximation
            ``dfs``        DFS-based local search improvement
            ``combined``   Best quality; combines both approaches
            ============== ===========================================

        seed : int, optional
            Random seed for reproducibility (default 0).
        eager_size : int, optional
            Maximum path length for the eager path search heuristic
            used by the ``"combined"`` algorithm (default 100).

        Returns
        -------
        EdgeOrientationResult
            ``max_out_degree`` -- the maximum out-degree achieved,
            ``out_degrees`` -- 1-D int array of out-degree per vertex,
            ``edge_heads`` -- 1-D int array giving the head (target) of
            each oriented edge (same length as ``g.adjncy``).

        Raises
        ------
        InvalidModeError
            If *algorithm* is not one of the valid choices.
        """
        from chszlablib._heiorient import orient_edges as _orient_edges

        if algorithm not in _VALID_ALGORITHMS:
            raise InvalidModeError(
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
