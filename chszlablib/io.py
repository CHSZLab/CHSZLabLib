"""METIS graph file I/O for CHSZLabLib."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def read_metis(path: Union[str, Path]) -> "Graph":
    """Read a graph from a METIS-format file.

    METIS format overview::

        % optional comment lines
        n m [fmt]
        [vwgt] neighbor1 [ewgt1] neighbor2 [ewgt2] ...
        ...

    ``fmt`` encodes which weights are present:

    - omitted or ``0``: no weights
    - ``1``:  edge weights only
    - ``10``: node weights only
    - ``11``: both node and edge weights

    Neighbors in the file are **1-indexed**; internally they are **0-indexed**.

    Parameters
    ----------
    path : str or Path
        Path to the METIS file.

    Returns
    -------
    Graph
        A finalized :class:`~chszlablib.graph.Graph`.
    """
    from chszlablib.graph import Graph

    path = Path(path)

    with open(path, "r") as fh:
        lines = fh.readlines()

    # Strip leading comments to find the header; keep blank lines (isolated nodes)
    header_found = False
    header_line = ""
    adj_lines_raw: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not header_found:
            if stripped == "" or stripped.startswith("%"):
                continue
            header_line = stripped
            header_found = True
        else:
            if stripped.startswith("%"):
                continue
            adj_lines_raw.append(stripped)

    if not header_found:
        raise ValueError("METIS file is empty or contains only comments")

    # Parse header
    header = header_line.split()
    n = int(header[0])
    m = int(header[1])
    fmt = int(header[2]) if len(header) >= 3 else 0

    has_node_weights = fmt in (10, 11)
    has_edge_weights = fmt in (1, 11)

    # Parse adjacency lines
    adj_lines = adj_lines_raw
    if len(adj_lines) != n:
        raise ValueError(
            f"Expected {n} adjacency lines, got {len(adj_lines)}"
        )

    # Build CSR directly
    xadj = np.empty(n + 1, dtype=np.int64)
    xadj[0] = 0

    # First pass: determine sizes
    all_neighbors: list[list[int]] = []
    all_eweights: list[list[int]] = []
    node_weights = np.ones(n, dtype=np.int64)

    for i, line in enumerate(adj_lines):
        tokens = list(map(int, line.split()))
        pos = 0

        if has_node_weights:
            node_weights[i] = tokens[pos]
            pos += 1

        neighbors: list[int] = []
        eweights: list[int] = []

        while pos < len(tokens):
            # neighbor is 1-indexed in file -> 0-indexed internally
            neighbor = tokens[pos] - 1
            pos += 1
            if has_edge_weights:
                ew = tokens[pos]
                pos += 1
            else:
                ew = 1
            neighbors.append(neighbor)
            eweights.append(ew)

        all_neighbors.append(neighbors)
        all_eweights.append(eweights)
        xadj[i + 1] = xadj[i] + len(neighbors)

    total = int(xadj[n])
    adjncy = np.empty(total, dtype=np.int32)
    edge_weights = np.empty(total, dtype=np.int64)

    idx = 0
    for neighbors, eweights in zip(all_neighbors, all_eweights):
        for nb, ew in zip(neighbors, eweights):
            adjncy[idx] = nb
            edge_weights[idx] = ew
            idx += 1

    return Graph.from_csr(
        xadj, adjncy,
        node_weights=node_weights if has_node_weights else None,
        edge_weights=edge_weights if has_edge_weights else None,
    )


def write_metis(graph: "Graph", path: Union[str, Path]) -> None:
    """Write a graph to a METIS-format file.

    Parameters
    ----------
    graph : Graph
        The graph to write. Must be finalized (or will be auto-finalized).
    path : str or Path
        Output file path.
    """
    path = Path(path)

    n = graph.num_nodes
    m = graph.num_edges

    # Determine fmt
    has_node_weights = not np.all(graph.node_weights == 1)
    has_edge_weights = not np.all(graph.edge_weights == 1)

    if has_node_weights and has_edge_weights:
        fmt = 11
    elif has_node_weights:
        fmt = 10
    elif has_edge_weights:
        fmt = 1
    else:
        fmt = 0

    with open(path, "w") as fh:
        # Header
        if fmt != 0:
            fh.write(f"{n} {m} {fmt}\n")
        else:
            fh.write(f"{n} {m}\n")

        # Adjacency lines
        xadj = graph.xadj
        adjncy = graph.adjncy
        nw = graph.node_weights
        ew = graph.edge_weights

        for i in range(n):
            parts: list[str] = []

            if has_node_weights:
                parts.append(str(int(nw[i])))

            start = int(xadj[i])
            end = int(xadj[i + 1])
            for j in range(start, end):
                # 0-indexed -> 1-indexed in file
                parts.append(str(int(adjncy[j]) + 1))
                if has_edge_weights:
                    parts.append(str(int(ew[j])))

            fh.write(" ".join(parts) + "\n")
