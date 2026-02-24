"""Graph data structure with CSR backend for CHSZLabLib."""

from __future__ import annotations

import numpy as np
from typing import Optional

from chszlablib.exceptions import InvalidGraphError, GraphNotFinalizedError


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
            raise InvalidGraphError(f"num_nodes must be non-negative, got {num_nodes}")
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
            raise GraphNotFinalizedError("Cannot add edges after finalize()")
        self._validate_node(u)
        self._validate_node(v)
        if u == v:
            raise InvalidGraphError(f"Self-loops are not allowed: ({u}, {v})")
        key = (min(u, v), max(u, v))
        if key in self._edge_set:
            raise InvalidGraphError(f"Duplicate edge: ({u}, {v})")
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
            raise GraphNotFinalizedError("Cannot set node weights after finalize()")
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
            raise InvalidGraphError(
                f"xadj must be a 1-D array of length >= 1, "
                f"got shape {xadj.shape}"
            )
        n = len(xadj) - 1
        if xadj[0] != 0:
            raise InvalidGraphError(f"xadj[0] must be 0, got {xadj[0]}")
        if xadj[-1] != len(adjncy):
            raise InvalidGraphError(
                f"xadj[-1] must equal len(adjncy) ({len(adjncy)}), "
                f"got {xadj[-1]}"
            )
        if not np.all(np.diff(xadj) >= 0):
            raise InvalidGraphError("xadj must be monotonically non-decreasing")
        if len(adjncy) > 0:
            if np.any(adjncy < 0) or np.any(adjncy >= n):
                raise InvalidGraphError(
                    f"adjncy values must be in [0, {n}), "
                    f"got range [{adjncy.min()}, {adjncy.max()}]"
                )
        if node_weights is not None:
            nw = np.asarray(node_weights)
            if nw.shape != (n,):
                raise InvalidGraphError(
                    f"node_weights must have shape ({n},), got {nw.shape}"
                )
        if edge_weights is not None:
            ew = np.asarray(edge_weights)
            if ew.shape != (len(adjncy),):
                raise InvalidGraphError(
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

    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[int, int]] | list[tuple[int, int, int]],
        num_nodes: int | None = None,
    ) -> Graph:
        """Construct a Graph from a list of edges.

        Parameters
        ----------
        edges : list of (u, v) or (u, v, weight) tuples
            Edge list. If tuples have 3 elements, the third is the edge weight.
        num_nodes : int or None
            Number of nodes. If ``None``, inferred as ``max(node index) + 1``.

        Returns
        -------
        Graph
            A finalized graph.
        """
        if not edges:
            n = num_nodes if num_nodes is not None else 0
            g = cls(num_nodes=n)
            g.finalize()
            return g

        if num_nodes is None:
            max_id = max(max(e[0], e[1]) for e in edges)
            num_nodes = max_id + 1

        g = cls(num_nodes=num_nodes)
        for edge in edges:
            if len(edge) == 3:
                g.add_edge(edge[0], edge[1], weight=edge[2])
            else:
                g.add_edge(edge[0], edge[1])
        g.finalize()
        return g

    @classmethod
    def from_networkx(cls, G) -> Graph:
        """Construct a Graph from a NetworkX graph.

        Parameters
        ----------
        G : networkx.Graph
            An undirected NetworkX graph. If edges have a ``"weight"``
            attribute, it is used as the edge weight.

        Returns
        -------
        Graph
            A finalized graph.

        Raises
        ------
        ImportError
            If NetworkX is not installed.
        TypeError
            If *G* is a directed graph.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for from_networkx(). "
                "Install it with: pip install networkx"
            )

        if isinstance(G, nx.DiGraph):
            raise TypeError(
                "from_networkx() requires an undirected graph, "
                "got a DiGraph"
            )

        A = nx.to_scipy_sparse_array(G, format="csr", weight="weight")
        return cls.from_scipy_sparse(A)

    @classmethod
    def from_scipy_sparse(cls, A) -> Graph:
        """Construct a Graph from a SciPy sparse CSR matrix.

        The matrix is interpreted as an adjacency matrix of an undirected
        graph. Only the upper triangle is read; the matrix need not be
        symmetric (the lower triangle is ignored).

        Parameters
        ----------
        A : scipy.sparse.csr_matrix or scipy.sparse.csr_array
            Sparse adjacency matrix in CSR format.

        Returns
        -------
        Graph
            A finalized graph.

        Raises
        ------
        ImportError
            If SciPy is not installed.
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError(
                "SciPy is required for from_scipy_sparse(). "
                "Install it with: pip install scipy"
            )

        if not sp.issparse(A):
            raise InvalidGraphError("Input must be a SciPy sparse matrix")

        A = sp.csr_array(A)

        xadj = np.asarray(A.indptr, dtype=np.int64)
        adjncy = np.asarray(A.indices, dtype=np.int32)

        if A.data is not None and not np.all(A.data == 1):
            edge_weights = np.asarray(A.data, dtype=np.int64)
        else:
            edge_weights = None

        return cls.from_csr(xadj, adjncy, edge_weights=edge_weights)

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

    def to_networkx(self):
        """Convert this graph to a NetworkX graph.

        Returns
        -------
        networkx.Graph
            An undirected NetworkX graph. Edge weights are included if
            any weight differs from 1. Node weights are stored as a
            ``"weight"`` node attribute.

        Raises
        ------
        ImportError
            If NetworkX is not installed.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for to_networkx(). "
                "Install it with: pip install networkx"
            )

        self.finalize()
        G = nx.Graph()
        G.add_nodes_from(range(self._num_nodes))

        has_ew = not np.all(self._edge_weights == 1)
        has_nw = not np.all(self._node_weights == 1)

        # Add edges (only one direction: u < v to avoid duplicates)
        for u in range(self._num_nodes):
            start = int(self._xadj[u])
            end = int(self._xadj[u + 1])
            for idx in range(start, end):
                v = int(self._adjncy[idx])
                if v > u:
                    if has_ew:
                        G.add_edge(u, v, weight=int(self._edge_weights[idx]))
                    else:
                        G.add_edge(u, v)

        if has_nw:
            for i in range(self._num_nodes):
                G.nodes[i]["weight"] = int(self._node_weights[i])

        return G

    def to_scipy_sparse(self):
        """Convert this graph to a SciPy CSR sparse matrix.

        Returns
        -------
        scipy.sparse.csr_array
            Sparse adjacency matrix.

        Raises
        ------
        ImportError
            If SciPy is not installed.
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError(
                "SciPy is required for to_scipy_sparse(). "
                "Install it with: pip install scipy"
            )

        self.finalize()
        return sp.csr_array(
            (self._edge_weights.copy(), self._adjncy.copy(), self._xadj.copy()),
            shape=(self._num_nodes, self._num_nodes),
        )

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
            raise InvalidGraphError(
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
