from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class MaxCutResult:
    """Result of a max-cut computation."""

    cut_value: int
    partition: np.ndarray | None


def maxcut(
    g: Graph,
    method: str = "heuristic",
    time_limit: float = 1.0,
) -> MaxCutResult:
    """Compute a maximum cut of a graph.

    Always applies FPT data-reduction rules (kernelization) first,
    then solves the reduced instance.

    Parameters
    ----------
    g : Graph
        Input undirected graph (may be weighted).
    method : str
        ``"heuristic"`` (default) -- fast heuristic solver.
        ``"exact"`` -- exact FPT algorithm (exponential in the size
        of the linear kernel's marked vertex set; only feasible for
        small instances or instances that kernelize well).
    time_limit : float
        Time limit in seconds (default 1.0).

    Returns
    -------
    MaxCutResult
        *cut_value* is the total weight of edges crossing the cut.
        *partition* is an array of 0/1 assignments per node.
    """
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
