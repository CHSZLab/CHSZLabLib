"""Maximum independent set algorithms via KaMIS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph


@dataclass
class MISResult:
    """Result of a maximum independent set computation."""

    size: int
    weight: int
    vertices: np.ndarray


def redumis(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
    full_kernelization: bool = False,
) -> MISResult:
    """Compute an unweighted MIS using the ReduMIS evolutionary algorithm.

    Parameters
    ----------
    g : Graph
        Input graph.
    time_limit : float
        Time limit in seconds (default 10.0).
    seed : int
        Random seed (default 0).
    full_kernelization : bool
        Use full kernelization (True) or FastKer (False, default).

    Returns
    -------
    MISResult
        Contains *size*, *weight*, and *vertices* array.
    """
    from chszlablib._kamis import redumis as _redumis

    g.finalize()
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

    is_size, is_verts = _redumis(xadj, adjncy, vwgt, time_limit, seed,
                                 full_kernelization)
    weight = int(np.sum(g.node_weights[is_verts])) if g.node_weights is not None and len(is_verts) > 0 else is_size
    return MISResult(size=is_size, weight=weight, vertices=is_verts)


def online_mis(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
    ils_iterations: int = 15000,
) -> MISResult:
    """Compute an unweighted MIS using the OnlineMIS local search algorithm.

    Parameters
    ----------
    g : Graph
        Input graph.
    time_limit : float
        Time limit in seconds (default 10.0).
    seed : int
        Random seed (default 0).
    ils_iterations : int
        Number of ILS iterations (default 15000).

    Returns
    -------
    MISResult
        Contains *size*, *weight*, and *vertices* array.
    """
    from chszlablib._kamis import online_mis as _online_mis

    g.finalize()
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

    is_size, is_verts = _online_mis(xadj, adjncy, vwgt, time_limit, seed,
                                    ils_iterations)
    weight = int(np.sum(g.node_weights[is_verts])) if g.node_weights is not None and len(is_verts) > 0 else is_size
    return MISResult(size=is_size, weight=weight, vertices=is_verts)


def branch_reduce(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
) -> MISResult:
    """Compute a weighted MIS using the exact branch-and-reduce algorithm.

    Parameters
    ----------
    g : Graph
        Input graph (node weights are used).
    time_limit : float
        Time limit in seconds (default 10.0).
    seed : int
        Random seed (default 0).

    Returns
    -------
    MISResult
        Contains *size*, *weight*, and *vertices* array.
    """
    from chszlablib._kamis_wmis import branch_reduce as _branch_reduce

    g.finalize()
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

    is_weight, is_verts = _branch_reduce(xadj, adjncy, vwgt, time_limit, seed)
    return MISResult(size=len(is_verts), weight=is_weight, vertices=is_verts)


def mmwis_solver(
    g: Graph,
    time_limit: float = 10.0,
    seed: int = 0,
) -> MISResult:
    """Compute a weighted MIS using the MMWIS memetic evolutionary algorithm.

    Parameters
    ----------
    g : Graph
        Input graph (node weights are used).
    time_limit : float
        Time limit in seconds (default 10.0).
    seed : int
        Random seed (default 0).

    Returns
    -------
    MISResult
        Contains *size*, *weight*, and *vertices* array.
    """
    from chszlablib._kamis_mmwis import mmwis_solver as _mmwis_solver

    g.finalize()
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

    is_weight, is_verts = _mmwis_solver(xadj, adjncy, vwgt, time_limit, seed)
    return MISResult(size=len(is_verts), weight=is_weight, vertices=is_verts)
