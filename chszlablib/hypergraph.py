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
