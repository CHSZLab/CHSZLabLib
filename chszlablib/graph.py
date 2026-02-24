"""Graph data structure with CSR backend for CHSZLabLib."""

from __future__ import annotations

import numpy as np
from typing import Optional


class Graph:
    """Undirected graph stored in Compressed Sparse Row (CSR) format.

    Build a graph incrementally via :meth:`add_edge` and :meth:`set_node_weight`,
    then call :meth:`finalize` (or access any property) to convert to CSR arrays.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph (fixed at construction time).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, num_nodes: int) -> None:
        if num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")
        self._num_nodes: int = num_nodes
        self._finalized: bool = False

        # Builder state (pre-finalize)
        self._edge_list: list[tuple[int, int, int]] = []  # (u, v, weight)
        self._edge_set: set[tuple[int, int]] = set()       # for duplicate detection
        self._node_weight_map: dict[int, int] = {}

        # CSR state (post-finalize)
        self._xadj: Optional[np.ndarray] = None
        self._adjncy: Optional[np.ndarray] = None
        self._node_weights: Optional[np.ndarray] = None
        self._edge_weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        """Add an undirected edge between nodes *u* and *v*.

        Parameters
        ----------
        u, v : int
            Node indices (0-based). Must be in range ``[0, num_nodes)``.
        weight : int, optional
            Edge weight (default 1).

        Raises
        ------
        RuntimeError
            If the graph has already been finalized.
        ValueError
            If *u* or *v* is out of range, if *u == v* (self-loop),
            or if the edge already exists.
        """
        if self._finalized:
            raise RuntimeError("Cannot add edges after finalize()")
        self._validate_node(u)
        self._validate_node(v)
        if u == v:
            raise ValueError(f"Self-loops are not allowed: ({u}, {v})")
        key = (min(u, v), max(u, v))
        if key in self._edge_set:
            raise ValueError(f"Duplicate edge: ({u}, {v})")
        self._edge_set.add(key)
        self._edge_list.append((u, v, weight))

    def set_node_weight(self, node: int, weight: int) -> None:
        """Set the weight of a node.

        Parameters
        ----------
        node : int
            Node index (0-based).
        weight : int
            Node weight.

        Raises
        ------
        RuntimeError
            If the graph has already been finalized.
        ValueError
            If *node* is out of range.
        """
        if self._finalized:
            raise RuntimeError("Cannot set node weights after finalize()")
        self._validate_node(node)
        self._node_weight_map[node] = weight

    def finalize(self) -> None:
        """Convert the edge list to CSR arrays.

        Idempotent: calling finalize() multiple times has no effect after
        the first call.
        """
        if self._finalized:
            return

        n = self._num_nodes

        # Build adjacency lists (both directions for undirected)
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for u, v, w in self._edge_list:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Sort adjacency lists by neighbor index for deterministic CSR
        for i in range(n):
            adj[i].sort(key=lambda t: t[0])

        # Build CSR arrays
        xadj = np.empty(n + 1, dtype=np.int64)
        xadj[0] = 0
        for i in range(n):
            xadj[i + 1] = xadj[i] + len(adj[i])

        total_entries = int(xadj[n])
        adjncy = np.empty(total_entries, dtype=np.int32)
        edge_weights = np.empty(total_entries, dtype=np.int64)

        idx = 0
        for i in range(n):
            for neighbor, w in adj[i]:
                adjncy[idx] = neighbor
                edge_weights[idx] = w
                idx += 1

        # Node weights
        node_weights = np.ones(n, dtype=np.int64)
        for node, w in self._node_weight_map.items():
            node_weights[node] = w

        self._xadj = xadj
        self._adjncy = adjncy
        self._edge_weights = edge_weights
        self._node_weights = node_weights
        self._finalized = True

        # Release builder state
        self._edge_list = []
        self._edge_set = set()
        self._node_weight_map = {}

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csr(
        cls,
        xadj: np.ndarray,
        adjncy: np.ndarray,
        node_weights: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None,
    ) -> Graph:
        """Construct a Graph directly from CSR arrays.

        Parameters
        ----------
        xadj : array-like, shape (n+1,)
            Index pointers into *adjncy*.
        adjncy : array-like
            Concatenated adjacency lists.
        node_weights : array-like, shape (n,), optional
            Node weights (default: all ones).
        edge_weights : array-like, shape (len(adjncy),), optional
            Edge weights (default: all ones).

        Returns
        -------
        Graph
            A finalized graph.
        """
        xadj = np.asarray(xadj, dtype=np.int64)
        adjncy = np.asarray(adjncy, dtype=np.int32)

        if xadj.ndim != 1 or len(xadj) < 1:
            raise ValueError(
                f"xadj must be a 1-D array of length >= 1, "
                f"got shape {xadj.shape}"
            )
        n = len(xadj) - 1
        if xadj[0] != 0:
            raise ValueError(f"xadj[0] must be 0, got {xadj[0]}")
        if xadj[-1] != len(adjncy):
            raise ValueError(
                f"xadj[-1] must equal len(adjncy) ({len(adjncy)}), "
                f"got {xadj[-1]}"
            )
        if not np.all(np.diff(xadj) >= 0):
            raise ValueError("xadj must be monotonically non-decreasing")
        if len(adjncy) > 0:
            if np.any(adjncy < 0) or np.any(adjncy >= n):
                raise ValueError(
                    f"adjncy values must be in [0, {n}), "
                    f"got range [{adjncy.min()}, {adjncy.max()}]"
                )
        if node_weights is not None:
            nw = np.asarray(node_weights)
            if nw.shape != (n,):
                raise ValueError(
                    f"node_weights must have shape ({n},), got {nw.shape}"
                )
        if edge_weights is not None:
            ew = np.asarray(edge_weights)
            if ew.shape != (len(adjncy),):
                raise ValueError(
                    f"edge_weights must have shape ({len(adjncy)},), "
                    f"got {ew.shape}"
                )

        g = cls.__new__(cls)
        g._num_nodes = n
        g._finalized = True
        g._edge_list = []
        g._edge_set = set()
        g._node_weight_map = {}

        g._xadj = xadj
        g._adjncy = adjncy

        if node_weights is not None:
            g._node_weights = np.asarray(node_weights, dtype=np.int64)
        else:
            g._node_weights = np.ones(n, dtype=np.int64)

        if edge_weights is not None:
            g._edge_weights = np.asarray(edge_weights, dtype=np.int64)
        else:
            g._edge_weights = np.ones(len(adjncy), dtype=np.int64)

        return g

    @classmethod
    def from_metis(cls, path: str) -> Graph:
        """Read a graph from a METIS file.

        Parameters
        ----------
        path : str
            Path to the METIS file.

        Returns
        -------
        Graph
            A finalized graph.
        """
        from chszlablib.io import read_metis
        return read_metis(path)

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def to_metis(self, path: str) -> None:
        """Write this graph to a METIS file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        from chszlablib.io import write_metis
        write_metis(self, path)

    # ------------------------------------------------------------------
    # Properties (auto-finalize on access)
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Number of undirected edges (= len(adjncy) / 2)."""
        self.finalize()
        return len(self._adjncy) // 2

    @property
    def xadj(self) -> np.ndarray:
        """CSR index pointer array, shape ``(num_nodes + 1,)``."""
        self.finalize()
        return self._xadj

    @property
    def adjncy(self) -> np.ndarray:
        """CSR adjacency array."""
        self.finalize()
        return self._adjncy

    @property
    def node_weights(self) -> np.ndarray:
        """Node weight array, shape ``(num_nodes,)``."""
        self.finalize()
        return self._node_weights

    @property
    def edge_weights(self) -> np.ndarray:
        """Edge weight array (same length as :attr:`adjncy`)."""
        self.finalize()
        return self._edge_weights

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_node(self, node: int) -> None:
        if node < 0 or node >= self._num_nodes:
            raise ValueError(
                f"Node index {node} out of range [0, {self._num_nodes})"
            )

    def __repr__(self) -> str:
        if self._finalized:
            has_nw = not np.all(self._node_weights == 1)
            has_ew = not np.all(self._edge_weights == 1)
            weighted = has_nw or has_ew
            return (
                f"Graph(n={self._num_nodes}, m={len(self._adjncy) // 2}, "
                f"weighted={weighted})"
            )
        else:
            return (
                f"Graph(n={self._num_nodes}, "
                f"edges_added={len(self._edge_list)}, finalized=False)"
            )
