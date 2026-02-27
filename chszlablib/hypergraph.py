"""HyperGraph data structure with dual CSR backend for CHSZLabLib."""

from __future__ import annotations

import numpy as np

from chszlablib.exceptions import InvalidHyperGraphError, GraphNotFinalizedError


class HyperGraph:
    """Hypergraph stored in dual Compressed Sparse Row (CSR) format.

    A hypergraph consists of vertices and hyperedges, where each hyperedge
    connects two or more vertices.  Internally the structure is stored as
    two CSR-like index arrays:

    * **edge-to-vertex** (``eptr`` / ``everts``): for each hyperedge, the
      sorted list of vertices it contains.
    * **vertex-to-edge** (``vptr`` / ``vedges``): for each vertex, the list
      of hyperedges it belongs to.

    Build a hypergraph incrementally via :meth:`add_to_edge` /
    :meth:`set_edge`, then call :meth:`finalize` (or access any property)
    to convert to CSR arrays.

    Parameters
    ----------
    num_nodes : int
        Number of vertices (fixed at construction time).
    num_edges : int
        Number of hyperedges (fixed at construction time).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, num_nodes: int, num_edges: int) -> None:
        if num_nodes < 0:
            raise InvalidHyperGraphError(
                f"num_nodes must be non-negative, got {num_nodes}"
            )
        if num_edges < 0:
            raise InvalidHyperGraphError(
                f"num_edges must be non-negative, got {num_edges}"
            )
        self._num_nodes: int = num_nodes
        self._num_edges: int = num_edges
        self._finalized: bool = False

        # Builder state (pre-finalize)
        self._edge_contents: list[list[int]] = [[] for _ in range(num_edges)]
        self._edge_vertex_sets: list[set[int]] = [set() for _ in range(num_edges)]
        self._node_weight_map: dict[int, int] = {}
        self._edge_weight_map: dict[int, int] = {}

        # CSR state (post-finalize)
        self._vptr: np.ndarray | None = None
        self._vedges: np.ndarray | None = None
        self._eptr: np.ndarray | None = None
        self._everts: np.ndarray | None = None
        self._node_weights: np.ndarray | None = None
        self._edge_weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_to_edge(self, edge_id: int, vertex: int) -> None:
        """Add a single vertex to a hyperedge.

        Parameters
        ----------
        edge_id : int
            Hyperedge index (0-based). Must be in range ``[0, num_edges)``.
        vertex : int
            Vertex index (0-based). Must be in range ``[0, num_nodes)``.

        Raises
        ------
        RuntimeError
            If the hypergraph has already been finalized.
        ValueError
            If *edge_id* or *vertex* is out of range, or *vertex* is
            already in the hyperedge.
        """
        if self._finalized:
            raise GraphNotFinalizedError(
                "Cannot modify hypergraph after finalize()"
            )
        self._validate_edge_id(edge_id)
        self._validate_vertex(vertex)
        if vertex in self._edge_vertex_sets[edge_id]:
            raise InvalidHyperGraphError(
                f"Duplicate vertex {vertex} in hyperedge {edge_id}"
            )
        self._edge_vertex_sets[edge_id].add(vertex)
        self._edge_contents[edge_id].append(vertex)

    def set_edge(self, edge_id: int, vertices: list[int]) -> None:
        """Set all vertices of a hyperedge at once.

        Replaces any previously added vertices for this edge.

        Parameters
        ----------
        edge_id : int
            Hyperedge index (0-based). Must be in range ``[0, num_edges)``.
        vertices : list[int]
            Vertex indices. Each must be in range ``[0, num_nodes)``.
            Duplicates are not allowed.

        Raises
        ------
        RuntimeError
            If the hypergraph has already been finalized.
        ValueError
            If *edge_id* is out of range, any vertex is out of range,
            or there are duplicate vertices.
        """
        if self._finalized:
            raise GraphNotFinalizedError(
                "Cannot modify hypergraph after finalize()"
            )
        self._validate_edge_id(edge_id)
        seen: set[int] = set()
        for v in vertices:
            self._validate_vertex(v)
            if v in seen:
                raise InvalidHyperGraphError(
                    f"Duplicate vertex {v} in hyperedge {edge_id}"
                )
            seen.add(v)
        self._edge_contents[edge_id] = list(vertices)
        self._edge_vertex_sets[edge_id] = seen

    def set_node_weight(self, node: int, weight: int) -> None:
        """Set the weight of a vertex.

        Parameters
        ----------
        node : int
            Vertex index (0-based).
        weight : int
            Vertex weight.

        Raises
        ------
        RuntimeError
            If the hypergraph has already been finalized.
        ValueError
            If *node* is out of range.
        """
        if self._finalized:
            raise GraphNotFinalizedError(
                "Cannot set node weights after finalize()"
            )
        self._validate_vertex(node)
        self._node_weight_map[node] = weight

    def set_edge_weight(self, edge: int, weight: int) -> None:
        """Set the weight of a hyperedge.

        Parameters
        ----------
        edge : int
            Hyperedge index (0-based).
        weight : int
            Hyperedge weight.

        Raises
        ------
        RuntimeError
            If the hypergraph has already been finalized.
        ValueError
            If *edge* is out of range.
        """
        if self._finalized:
            raise GraphNotFinalizedError(
                "Cannot set edge weights after finalize()"
            )
        self._validate_edge_id(edge)
        self._edge_weight_map[edge] = weight

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Build dual CSR arrays from the builder state.

        Idempotent: calling finalize() multiple times has no effect after
        the first call.

        Raises
        ------
        ValueError
            If any hyperedge contains fewer than 1 vertex.
        """
        if self._finalized:
            return

        n = self._num_nodes
        m = self._num_edges

        # Validate: every edge must have >= 1 vertex
        for eid in range(m):
            if len(self._edge_contents[eid]) == 0:
                raise InvalidHyperGraphError(
                    f"Hyperedge {eid} has no vertices (must have >= 1)"
                )

        # Sort vertices within each edge for deterministic output
        for eid in range(m):
            self._edge_contents[eid].sort()

        # Build eptr / everts (edge-to-vertex CSR)
        eptr = np.empty(m + 1, dtype=np.int64)
        eptr[0] = 0
        for eid in range(m):
            eptr[eid + 1] = eptr[eid] + len(self._edge_contents[eid])

        total_ev = int(eptr[m]) if m > 0 else 0
        everts = np.empty(total_ev, dtype=np.int32)
        idx = 0
        for eid in range(m):
            for v in self._edge_contents[eid]:
                everts[idx] = v
                idx += 1

        # Build vptr / vedges (vertex-to-edge CSR)
        vertex_edge_lists: list[list[int]] = [[] for _ in range(n)]
        for eid in range(m):
            for v in self._edge_contents[eid]:
                vertex_edge_lists[v].append(eid)

        vptr = np.empty(n + 1, dtype=np.int64)
        vptr[0] = 0
        for vid in range(n):
            vptr[vid + 1] = vptr[vid] + len(vertex_edge_lists[vid])

        total_ve = int(vptr[n]) if n > 0 else 0
        vedges = np.empty(total_ve, dtype=np.int32)
        idx = 0
        for vid in range(n):
            for eid in vertex_edge_lists[vid]:
                vedges[idx] = eid
                idx += 1

        # Build weight arrays
        node_weights = np.ones(n, dtype=np.int64)
        for node, w in self._node_weight_map.items():
            node_weights[node] = w

        edge_weights = np.ones(m, dtype=np.int64)
        for eid, w in self._edge_weight_map.items():
            edge_weights[eid] = w

        self._vptr = vptr
        self._vedges = vedges
        self._eptr = eptr
        self._everts = everts
        self._node_weights = node_weights
        self._edge_weights = edge_weights
        self._finalized = True

        # Release builder state
        self._edge_contents = []
        self._edge_vertex_sets = []
        self._node_weight_map = {}
        self._edge_weight_map = {}

    # ------------------------------------------------------------------
    # Properties (auto-finalize on access)
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of vertices in the hypergraph."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Number of hyperedges in the hypergraph."""
        return self._num_edges

    @property
    def eptr(self) -> np.ndarray:
        """Edge-to-vertex index pointer array, shape ``(num_edges + 1,)``."""
        self.finalize()
        return self._eptr

    @property
    def everts(self) -> np.ndarray:
        """Edge-to-vertex array (concatenated sorted vertex IDs per edge)."""
        self.finalize()
        return self._everts

    @property
    def vptr(self) -> np.ndarray:
        """Vertex-to-edge index pointer array, shape ``(num_nodes + 1,)``."""
        self.finalize()
        return self._vptr

    @property
    def vedges(self) -> np.ndarray:
        """Vertex-to-edge array (concatenated edge IDs per vertex)."""
        self.finalize()
        return self._vedges

    @property
    def node_weights(self) -> np.ndarray:
        """Node weight array, shape ``(num_nodes,)``."""
        self.finalize()
        return self._node_weights

    @property
    def edge_weights(self) -> np.ndarray:
        """Edge weight array, shape ``(num_edges,)``."""
        self.finalize()
        return self._edge_weights

    # ------------------------------------------------------------------
    # Class methods (batch constructors)
    # ------------------------------------------------------------------

    @classmethod
    def from_edge_list(
        cls,
        edges: list[list[int]],
        num_nodes: int | None = None,
        node_weights: list[int] | np.ndarray | None = None,
        edge_weights: list[int] | np.ndarray | None = None,
    ) -> HyperGraph:
        """Construct a HyperGraph from a list of hyperedges.

        Parameters
        ----------
        edges : list of list of int
            Each inner list is a hyperedge containing vertex IDs.
        num_nodes : int or None
            Number of vertices. If ``None``, inferred as
            ``max(vertex ID) + 1``.
        node_weights : array-like, shape (num_nodes,), optional
            Vertex weights. If ``None``, all weights default to 1.
        edge_weights : array-like, shape (num_edges,), optional
            Hyperedge weights. If ``None``, all weights default to 1.

        Returns
        -------
        HyperGraph
            A finalized hypergraph.
        """
        m = len(edges)

        if num_nodes is None:
            if m == 0:
                num_nodes = 0
            else:
                max_id = max(
                    (v for edge in edges for v in edge),
                    default=-1,
                )
                num_nodes = max_id + 1

        hg = cls(num_nodes=num_nodes, num_edges=m)

        for eid, verts in enumerate(edges):
            hg.set_edge(eid, verts)

        if node_weights is not None:
            nw = np.asarray(node_weights)
            for i in range(len(nw)):
                hg.set_node_weight(i, int(nw[i]))

        if edge_weights is not None:
            ew = np.asarray(edge_weights)
            for i in range(len(ew)):
                hg.set_edge_weight(i, int(ew[i]))

        hg.finalize()
        return hg

    @classmethod
    def from_dual_csr(
        cls,
        vptr: np.ndarray,
        vedges: np.ndarray,
        eptr: np.ndarray,
        everts: np.ndarray,
        node_weights: np.ndarray | None = None,
        edge_weights: np.ndarray | None = None,
    ) -> HyperGraph:
        """Construct a HyperGraph directly from dual CSR arrays.

        Parameters
        ----------
        vptr : array-like, shape (n+1,)
            Vertex-to-edge index pointer array.
        vedges : array-like
            Vertex-to-edge array (concatenated edge IDs per vertex).
        eptr : array-like, shape (m+1,)
            Edge-to-vertex index pointer array.
        everts : array-like
            Edge-to-vertex array (concatenated vertex IDs per edge).
        node_weights : array-like, shape (n,), optional
            Vertex weights (default: all ones).
        edge_weights : array-like, shape (m,), optional
            Hyperedge weights (default: all ones).

        Returns
        -------
        HyperGraph
            A finalized hypergraph.

        Raises
        ------
        InvalidHyperGraphError
            If the CSR arrays are structurally invalid.
        """
        eptr = np.asarray(eptr, dtype=np.int64)
        everts = np.asarray(everts, dtype=np.int32)
        vptr = np.asarray(vptr, dtype=np.int64)
        vedges = np.asarray(vedges, dtype=np.int32)

        # Validate eptr
        if eptr.ndim != 1 or len(eptr) < 1:
            raise InvalidHyperGraphError(
                f"eptr must be a 1-D array of length >= 1, "
                f"got shape {eptr.shape}"
            )
        if eptr[0] != 0:
            raise InvalidHyperGraphError(
                f"eptr[0] must be 0, got {eptr[0]}"
            )
        if eptr[-1] != len(everts):
            raise InvalidHyperGraphError(
                f"eptr[-1] must equal len(everts) ({len(everts)}), "
                f"got {eptr[-1]}"
            )
        if not np.all(np.diff(eptr) >= 0):
            raise InvalidHyperGraphError(
                "eptr must be monotonically non-decreasing"
            )

        # Validate vptr
        if vptr.ndim != 1 or len(vptr) < 1:
            raise InvalidHyperGraphError(
                f"vptr must be a 1-D array of length >= 1, "
                f"got shape {vptr.shape}"
            )
        if vptr[0] != 0:
            raise InvalidHyperGraphError(
                f"vptr[0] must be 0, got {vptr[0]}"
            )
        if vptr[-1] != len(vedges):
            raise InvalidHyperGraphError(
                f"vptr[-1] must equal len(vedges) ({len(vedges)}), "
                f"got {vptr[-1]}"
            )
        if not np.all(np.diff(vptr) >= 0):
            raise InvalidHyperGraphError(
                "vptr must be monotonically non-decreasing"
            )

        n = len(vptr) - 1
        m = len(eptr) - 1

        # Validate bounds on everts
        if len(everts) > 0:
            if np.any(everts < 0) or np.any(everts >= n):
                raise InvalidHyperGraphError(
                    f"everts values must be in [0, {n}), "
                    f"got range [{everts.min()}, {everts.max()}]"
                )

        # Validate bounds on vedges
        if len(vedges) > 0:
            if np.any(vedges < 0) or np.any(vedges >= m):
                raise InvalidHyperGraphError(
                    f"vedges values must be in [0, {m}), "
                    f"got range [{vedges.min()}, {vedges.max()}]"
                )

        # Validate weights shapes
        if node_weights is not None:
            nw = np.asarray(node_weights)
            if nw.shape != (n,):
                raise InvalidHyperGraphError(
                    f"node_weights must have shape ({n},), got {nw.shape}"
                )

        if edge_weights is not None:
            ew = np.asarray(edge_weights)
            if ew.shape != (m,):
                raise InvalidHyperGraphError(
                    f"edge_weights must have shape ({m},), got {ew.shape}"
                )

        # Build the object bypassing __init__
        hg = cls.__new__(cls)
        hg._num_nodes = n
        hg._num_edges = m
        hg._finalized = True

        # Builder state (empty, already finalized)
        hg._edge_contents = []
        hg._edge_vertex_sets = []
        hg._node_weight_map = {}
        hg._edge_weight_map = {}

        # CSR state
        hg._vptr = vptr
        hg._vedges = vedges
        hg._eptr = eptr
        hg._everts = everts

        if node_weights is not None:
            hg._node_weights = np.asarray(node_weights, dtype=np.int64)
        else:
            hg._node_weights = np.ones(n, dtype=np.int64)

        if edge_weights is not None:
            hg._edge_weights = np.asarray(edge_weights, dtype=np.int64)
        else:
            hg._edge_weights = np.ones(m, dtype=np.int64)

        return hg

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_graph(self) -> "Graph":
        """Convert this hypergraph to a graph via clique expansion.

        Each hyperedge is expanded into a clique: for a hyperedge containing
        vertices {v1, v2, ..., vk}, edges (vi, vj) are added for all i < j.
        Duplicate edges from overlapping hyperedges are added only once.
        Node weights are preserved; edge weights are set to 1.

        Returns
        -------
        Graph
            A finalized graph.
        """
        from chszlablib.graph import Graph

        self.finalize()
        g = Graph(self._num_nodes)

        # Copy node weights
        for i in range(self._num_nodes):
            w = int(self._node_weights[i])
            if w != 1:
                g.set_node_weight(i, w)

        # Clique expansion
        seen: set[tuple[int, int]] = set()
        for e in range(self._num_edges):
            start = int(self._eptr[e])
            end = int(self._eptr[e + 1])
            verts = sorted(int(v) for v in self._everts[start:end])
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    key = (verts[i], verts[j])
                    if key not in seen:
                        seen.add(key)
                        g.add_edge(verts[i], verts[j])

        g.finalize()
        return g

    # ------------------------------------------------------------------
    # hMETIS I/O convenience methods
    # ------------------------------------------------------------------

    @classmethod
    def from_hmetis(cls, path: str) -> HyperGraph:
        """Read a hypergraph from an hMETIS-format file.

        Parameters
        ----------
        path : str
            Path to the hMETIS file.

        Returns
        -------
        HyperGraph
            A finalized hypergraph.

        See Also
        --------
        chszlablib.io.read_hmetis : The underlying reader.
        """
        from chszlablib.io import read_hmetis

        return read_hmetis(path)

    def to_hmetis(self, path: str) -> None:
        """Write this hypergraph to an hMETIS-format file.

        Parameters
        ----------
        path : str
            Output file path.

        See Also
        --------
        chszlablib.io.write_hmetis : The underlying writer.
        """
        from chszlablib.io import write_hmetis

        write_hmetis(self, path)

    # ------------------------------------------------------------------
    # Binary I/O
    # ------------------------------------------------------------------

    def save_binary(self, path: str) -> None:
        """Save this hypergraph to a binary file (NumPy npz format).

        The binary format stores the dual CSR arrays directly and loads
        approximately 100x faster than parsing an hMETIS text file.

        Parameters
        ----------
        path : str or Path
            Output file path.  By convention, use the ``.npz`` extension.
        """
        self.finalize()
        np.savez(
            path,
            __format_version__=np.array([1], dtype=np.int32),
            __format_type__=np.array([2], dtype=np.int32),
            eptr=self._eptr,
            everts=self._everts,
            vptr=self._vptr,
            vedges=self._vedges,
            node_weights=self._node_weights,
            edge_weights=self._edge_weights,
        )

    @classmethod
    def load_binary(cls, path: str) -> HyperGraph:
        """Load a hypergraph from a binary file (NumPy npz format).

        Parameters
        ----------
        path : str or Path
            Path to a ``.npz`` file previously created by :meth:`save_binary`.

        Returns
        -------
        HyperGraph
            A finalized hypergraph.

        Raises
        ------
        ValueError
            If the file does not contain a valid hypergraph or has an
            unsupported format version.
        """
        data = np.load(path, allow_pickle=False)

        version = int(data["__format_version__"][0]) if "__format_version__" in data else 0
        if version > 1:
            raise ValueError(
                f"Unsupported hypergraph binary format version {version}. "
                f"Please update CHSZLabLib."
            )

        if "__format_type__" in data:
            fmt_type = int(data["__format_type__"][0])
            if fmt_type != 2:
                raise ValueError(
                    f"Expected binary type 2 (hypergraph), got {fmt_type}"
                )

        return cls.from_dual_csr(
            vptr=data["vptr"],
            vedges=data["vedges"],
            eptr=data["eptr"],
            everts=data["everts"],
            node_weights=data.get("node_weights", None),
            edge_weights=data.get("edge_weights", None),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_vertex(self, vertex: int) -> None:
        if vertex < 0 or vertex >= self._num_nodes:
            raise InvalidHyperGraphError(
                f"Vertex index {vertex} out of range [0, {self._num_nodes})"
            )

    def _validate_edge_id(self, edge_id: int) -> None:
        if edge_id < 0 or edge_id >= self._num_edges:
            raise InvalidHyperGraphError(
                f"Edge index {edge_id} out of range [0, {self._num_edges})"
            )

    def __repr__(self) -> str:
        if self._finalized:
            has_nw = not np.all(self._node_weights == 1)
            has_ew = not np.all(self._edge_weights == 1)
            weighted = has_nw or has_ew
            return (
                f"HyperGraph(n={self._num_nodes}, m={self._num_edges}, "
                f"weighted={weighted})"
            )
        else:
            return (
                f"HyperGraph(n={self._num_nodes}, m={self._num_edges}, "
                f"finalized=False)"
            )
