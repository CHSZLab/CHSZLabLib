"""Maximum weight independent set via CHILS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class MWISResult:
    """Result of a maximum weight independent set computation."""

    weight: int
    vertices: np.ndarray


def mwis(
    g: Graph,
    time_limit: float = 10.0,
    num_concurrent: int = 4,
    seed: int = 0,
) -> MWISResult:
    """Compute a maximum weight independent set using CHILS.

    Parameters
    ----------
    g : Graph
        Input graph.
    time_limit : float
        Time limit in seconds (default 10.0).
    num_concurrent : int
        Number of concurrent solutions (default 4).
    seed : int
        Random seed (default 0).

    Returns
    -------
    MWISResult
        Contains *weight* and *vertices* array.
    """
    from chszlablib._chils import mwis as _mwis

    g.finalize()
    xadj = g.xadj.astype(np.int64, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    weights = g.node_weights.astype(np.int64, copy=False)

    total_weight, vertices = _mwis(
        xadj, adjncy, weights, time_limit, num_concurrent, seed,
    )
    return MWISResult(weight=int(total_weight), vertices=vertices)
