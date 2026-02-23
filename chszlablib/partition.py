"""Graph partitioning, node separators, and nested dissection via KaHIP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chszlablib.graph import Graph

_MODE_MAP = {
    "fast": 0,
    "eco": 1,
    "strong": 2,
    "fastsocial": 3,
    "ecosocial": 4,
    "strongsocial": 5,
}


@dataclass
class PartitionResult:
    """Result of a graph partitioning call."""

    edgecut: int
    assignment: np.ndarray


@dataclass
class SeparatorResult:
    """Result of a node separator computation."""

    num_separator_vertices: int
    separator: np.ndarray


@dataclass
class OrderingResult:
    """Result of a nested dissection ordering."""

    ordering: np.ndarray


def partition(
    g: Graph,
    num_parts: int = 2,
    mode: str = "eco",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
) -> PartitionResult:
    """Partition a graph into *num_parts* blocks using KaHIP.

    Parameters
    ----------
    g : Graph
        Input graph.
    num_parts : int
        Number of partitions (default 2).
    mode : str
        Quality preset: ``"fast"``, ``"eco"``, ``"strong"``,
        ``"fastsocial"``, ``"ecosocial"``, or ``"strongsocial"``.
    imbalance : float
        Allowed imbalance fraction (default 0.03 = 3 %).
    seed : int
        Random seed (default 0).
    suppress_output : bool
        Suppress KaHIP console output (default ``True``).

    Returns
    -------
    PartitionResult
        Contains *edgecut* and *assignment* array of length ``num_nodes``.
    """
    from chszlablib._kahip import kaffpa

    g.finalize()
    vwgt = g.node_weights.astype(np.int32, copy=False)
    xadj = g.xadj.astype(np.int32, copy=False)
    adjcwgt = g.edge_weights.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    edgecut, part = kaffpa(
        vwgt, xadj, adjcwgt, adjncy,
        num_parts, imbalance, suppress_output, seed,
        _MODE_MAP[mode.lower()],
    )
    return PartitionResult(edgecut=edgecut, assignment=part)


def node_separator(
    g: Graph,
    num_parts: int = 2,
    mode: str = "eco",
    imbalance: float = 0.03,
    seed: int = 0,
    suppress_output: bool = True,
) -> SeparatorResult:
    """Compute a node separator using KaHIP.

    Parameters
    ----------
    g : Graph
        Input graph.
    num_parts : int
        Number of partitions (default 2).
    mode : str
        Quality preset (see :func:`partition`).
    imbalance : float
        Allowed imbalance fraction (default 0.03).
    seed : int
        Random seed (default 0).
    suppress_output : bool
        Suppress KaHIP console output (default ``True``).

    Returns
    -------
    SeparatorResult
        Contains *num_separator_vertices* and *separator* array.
    """
    from chszlablib._kahip import node_separator as _ns

    g.finalize()
    vwgt = g.node_weights.astype(np.int32, copy=False)
    xadj = g.xadj.astype(np.int32, copy=False)
    adjcwgt = g.edge_weights.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    num_sep, sep = _ns(
        vwgt, xadj, adjcwgt, adjncy,
        num_parts, imbalance, suppress_output, seed,
        _MODE_MAP[mode.lower()],
    )
    return SeparatorResult(num_separator_vertices=num_sep, separator=sep)


def node_ordering(
    g: Graph,
    mode: str = "eco",
    seed: int = 0,
    suppress_output: bool = True,
) -> OrderingResult:
    """Compute a reduced nested dissection ordering using KaHIP.

    Parameters
    ----------
    g : Graph
        Input graph (edge weights are ignored).
    mode : str
        Quality preset (see :func:`partition`).
    seed : int
        Random seed (default 0).
    suppress_output : bool
        Suppress KaHIP console output (default ``True``).

    Returns
    -------
    OrderingResult
        Contains *ordering* permutation array of length ``num_nodes``.
    """
    from chszlablib._kahip import node_ordering as _no

    g.finalize()
    xadj = g.xadj.astype(np.int32, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)

    ordering = _no(
        xadj, adjncy,
        suppress_output, seed,
        _MODE_MAP[mode.lower()],
    )
    return OrderingResult(ordering=ordering)
