"""Global minimum cut via VieCut algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph

_ALGO_MAP = {
    "viecut": "vc",
    "vc": "vc",
    "noi": "noi",
    "ks": "ks",
    "matula": "matula",
    "pr": "pr",
    "cactus": "cactus",
}


@dataclass
class MincutResult:
    """Result of a global minimum cut computation.

    Attributes
    ----------
    cut_value : int
        The weight of the minimum cut.
    partition : np.ndarray
        A 0/1 array of length ``num_nodes`` indicating which side of the
        cut each node belongs to.
    """

    cut_value: int
    partition: np.ndarray


def mincut(g: Graph, algorithm: str = "viecut", seed: int = 0) -> MincutResult:
    """Compute a global minimum cut of an undirected graph.

    Parameters
    ----------
    g : Graph
        The input graph.
    algorithm : str
        Algorithm to use. One of ``"viecut"`` (alias ``"vc"``), ``"noi"``,
        ``"ks"``, ``"matula"``, ``"pr"``, ``"cactus"``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MincutResult
        Object with ``cut_value`` and ``partition`` attributes.

    Raises
    ------
    ValueError
        If *algorithm* is not recognized.
    """
    from chszlablib._viecut import minimum_cut

    algo_key = algorithm.lower()
    if algo_key not in _ALGO_MAP:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. "
            f"Choose from: {', '.join(sorted(_ALGO_MAP))}"
        )

    g.finalize()

    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    adjwgt = g.edge_weights.astype(np.int32, copy=False)

    cut_value, partition = minimum_cut(
        xadj, adjncy, adjwgt,
        algorithm=_ALGO_MAP[algo_key],
        save_cut=True,
        seed=seed,
    )

    return MincutResult(cut_value=int(cut_value), partition=partition)
