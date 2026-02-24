from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from chszlablib.graph import Graph


@dataclass
class BMatchingResult:
    """Result of a b-matching computation."""

    weight: int
    size: int
    matched_edges: np.ndarray


def bmatching(
    g: Graph,
    capacities: Union[int, np.ndarray] = 1,
    algorithm: str = "ils",
    time_limit: float = 10.0,
    seed: int = 0,
    max_tries: int = 15,
) -> BMatchingResult:
    """Compute a maximum-weight b-matching on a standard graph.

    Find a maximum-weight set of edges such that each node is incident
    to at most *b* matched edges, where *b* is the node's capacity.

    Parameters
    ----------
    g : Graph
        Input undirected graph.
    capacities : int or ndarray
        Per-node capacities.  An ``int`` sets uniform capacity for all nodes;
        an array of length ``num_nodes`` sets individual capacities.
    algorithm : str
        Algorithm to use (default ``"ils"``).

        * ``"greedy"`` -- weight-ordered greedy heuristic (fast).
        * ``"ils"`` -- greedy + iterated local search (better quality).
        * ``"reduced"`` -- graph reductions + greedy.
        * ``"reduced+ils"`` -- reductions + greedy + ILS (best quality).
    time_limit : float
        Time limit in seconds for ILS (default 10.0).
    seed : int
        Random seed (default 0).
    max_tries : int
        ILS iterations without improvement before stopping (default 15).

    Returns
    -------
    BMatchingResult
        Contains *weight*, *size*, and *matched_edges* (indices of matched
        edges in CSR unique-edge order, i.e. edges where ``u < v``).
    """
    from chszlablib._bmatching import bmatching as _bmatching

    g.finalize()

    xadj = g.xadj.astype(np.int64, copy=False)
    adjncy = g.adjncy.astype(np.int32, copy=False)
    adjwgt = g.edge_weights.astype(np.int64, copy=False)

    n = g.num_nodes
    if isinstance(capacities, (int, np.integer)):
        cap = np.full(n, int(capacities), dtype=np.int32)
    else:
        cap = np.asarray(capacities, dtype=np.int32)
        if cap.shape != (n,):
            raise ValueError(
                f"capacities array must have shape ({n},), got {cap.shape}"
            )

    weight, size, matched = _bmatching(
        xadj, adjncy, adjwgt, cap, algorithm, time_limit, seed, max_tries,
    )

    return BMatchingResult(
        weight=int(weight),
        size=int(size),
        matched_edges=matched,
    )


def hypergraph_bmatching(
    num_nodes: int,
    edges: list[list[int]],
    capacities: Union[int, np.ndarray] = 1,
    edge_weights: np.ndarray | None = None,
    algorithm: str = "ils",
    time_limit: float = 10.0,
    seed: int = 0,
    max_tries: int = 15,
) -> BMatchingResult:
    """Compute a maximum-weight b-matching on a hypergraph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the hypergraph.
    edges : list of list of int
        Each inner list contains the node indices of a hyperedge.
    capacities : int or ndarray
        Per-node capacities.  An ``int`` sets uniform capacity; an array
        of length *num_nodes* sets individual capacities.
    edge_weights : ndarray, optional
        Weight of each hyperedge (default: all 1).
    algorithm : str
        Same choices as :func:`bmatching`.
    time_limit : float
        Time limit in seconds for ILS (default 10.0).
    seed : int
        Random seed (default 0).
    max_tries : int
        ILS iterations without improvement before stopping (default 15).

    Returns
    -------
    BMatchingResult
        Contains *weight*, *size*, and *matched_edges* (indices into
        the input *edges* list).
    """
    from chszlablib._bmatching import hypergraph_bmatching as _hg_bmatching

    num_edges = len(edges)

    # Build offset/pin arrays from list-of-lists
    offsets = np.empty(num_edges + 1, dtype=np.int64)
    offsets[0] = 0
    for i, e in enumerate(edges):
        offsets[i + 1] = offsets[i] + len(e)
    total_pins = int(offsets[num_edges])

    pins = np.empty(total_pins, dtype=np.int32)
    idx = 0
    for e in edges:
        for node in e:
            pins[idx] = node
            idx += 1

    if edge_weights is None:
        ew = np.ones(num_edges, dtype=np.int32)
    else:
        ew = np.asarray(edge_weights, dtype=np.int32)
        if ew.shape != (num_edges,):
            raise ValueError(
                f"edge_weights must have shape ({num_edges},), got {ew.shape}"
            )

    if isinstance(capacities, (int, np.integer)):
        cap = np.full(num_nodes, int(capacities), dtype=np.int32)
    else:
        cap = np.asarray(capacities, dtype=np.int32)
        if cap.shape != (num_nodes,):
            raise ValueError(
                f"capacities array must have shape ({num_nodes},), got {cap.shape}"
            )

    weight, size, matched = _hg_bmatching(
        num_nodes, offsets, pins, ew, cap,
        algorithm, time_limit, seed, max_tries,
    )

    return BMatchingResult(
        weight=int(weight),
        size=int(size),
        matched_edges=matched,
    )
