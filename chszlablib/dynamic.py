"""Dynamic graph algorithms: edge orientation, matching, and weighted MIS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DynOrientationResult:
    """Result of a dynamic edge orientation query."""

    max_out_degree: int
    """Maximum out-degree across all nodes."""
    out_degrees: np.ndarray
    """Out-degree array for all nodes (int32)."""


@dataclass
class DynMatchingResult:
    """Result of a dynamic matching query."""

    matching_size: int
    """Number of matched edges."""
    matching: np.ndarray
    """Matching array: matching[v] = mate of v, or -1 if unmatched (int32)."""


@dataclass
class DynWMISResult:
    """Result of a dynamic weighted MIS query."""

    weight: int
    """Total weight of the independent set."""
    vertices: np.ndarray
    """Boolean array: True if vertex is in the independent set."""


class DynEdgeOrientation:
    """Dynamic edge orientation (exact algorithms).

    Maintains an orientation of edges such that the maximum out-degree
    is minimized. Edges can be inserted and deleted incrementally.

    Parameters
    ----------
    num_nodes : int
        Number of vertices.
    algorithm : str, optional
        Algorithm name (default ``"kflips"``).
    seed : int, optional
        Random seed (default 0).
    """

    ALGORITHMS: tuple[str, ...] = (
        "bfs", "naive_opt", "impro_opt", "kflips", "rwalk", "naive",
        "brodal_fagerberg", "max_descending", "strong_opt", "strong_opt_dfs",
        "improved_opt", "improved_opt_dfs",
    )
    """Valid algorithm names."""

    DEFAULT_ALGORITHM: str = "kflips"

    def __init__(
        self,
        num_nodes: int,
        algorithm: str = "kflips",
        seed: int = 0,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. Valid: {self.ALGORITHMS}"
            )
        self._num_nodes = num_nodes
        self._algorithm = algorithm

        from chszlablib._dyn_orientation import DynOrientationSolver
        self._solver = DynOrientationSolver(num_nodes, algorithm, seed)

    def insert_edge(self, u: int, v: int) -> None:
        """Insert an undirected edge (u, v)."""
        self._solver.insert_edge(u, v)

    def delete_edge(self, u: int, v: int) -> None:
        """Delete an undirected edge (u, v)."""
        self._solver.delete_edge(u, v)

    def get_current_solution(self) -> DynOrientationResult:
        """Return the current edge orientation solution.

        Returns
        -------
        DynOrientationResult
            ``max_out_degree`` and ``out_degrees`` array.
        """
        return DynOrientationResult(
            max_out_degree=self._solver.get_max_out_degree(),
            out_degrees=self._solver.get_out_degrees(),
        )


class DynDeltaApproxOrientation:
    """Dynamic edge orientation (approximate algorithms).

    Maintains an approximate orientation of edges with bounded
    maximum out-degree. Edges can be inserted and deleted incrementally.

    Parameters
    ----------
    num_nodes : int
        Number of vertices.
    num_edges_hint : int, optional
        Hint for maximum number of edges (default 0, used for memory
        pre-allocation in CCHHQRS variants).
    algorithm : str, optional
        Algorithm name (default ``"improved_bfs"``).
    lambda_param : float, optional
        Lambda parameter for CCHHQRS variants (default 0.1).
    theta : int, optional
        Theta parameter for CCHHQRS variants (default 0).
    b : int, optional
        Fractional edge parameter for CCHHQRS variants (default 1).
    bfs_depth : int, optional
        BFS depth for BFS-based algorithms (default 20).
    """

    ALGORITHMS: tuple[str, ...] = (
        "cchhqrs", "limited_bfs", "strong_bfs", "improved_bfs",
        "packed_cchhqrs", "packed_cchhqrs_list", "packed_cchhqrs_map",
    )
    """Valid algorithm names."""

    DEFAULT_ALGORITHM: str = "improved_bfs"

    def __init__(
        self,
        num_nodes: int,
        num_edges_hint: int = 0,
        algorithm: str = "improved_bfs",
        lambda_param: float = 0.1,
        theta: int = 0,
        b: int = 1,
        bfs_depth: int = 20,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. Valid: {self.ALGORITHMS}"
            )
        self._num_nodes = num_nodes
        self._algorithm = algorithm

        from chszlablib._dyn_delta_approx import DynDeltaApproxSolver
        self._solver = DynDeltaApproxSolver(
            num_nodes, num_edges_hint, algorithm,
            lambda_param, theta, b, bfs_depth,
        )

    def insert_edge(self, u: int, v: int) -> None:
        """Insert an undirected edge (u, v)."""
        self._solver.insert_edge(u, v)

    def delete_edge(self, u: int, v: int) -> None:
        """Delete an undirected edge (u, v)."""
        self._solver.delete_edge(u, v)

    def get_current_solution(self) -> int:
        """Return the current maximum out-degree.

        Returns
        -------
        int
            The maximum out-degree in the current orientation.
        """
        return self._solver.get_max_out_degree()


class DynMatching:
    """Dynamic graph matching.

    Maintains a matching on a dynamic graph where edges can be
    inserted and deleted incrementally.

    Parameters
    ----------
    num_nodes : int
        Number of vertices.
    algorithm : str, optional
        Algorithm name (default ``"blossom"``).
    seed : int, optional
        Random seed (default 0).
    """

    ALGORITHMS: tuple[str, ...] = (
        "random_walk", "baswana_gupta_sen", "neiman_solomon",
        "naive", "blossom", "blossom_naive", "static_blossom",
    )
    """Valid algorithm names."""

    DEFAULT_ALGORITHM: str = "blossom"

    def __init__(
        self,
        num_nodes: int,
        algorithm: str = "blossom",
        seed: int = 0,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. Valid: {self.ALGORITHMS}"
            )
        self._num_nodes = num_nodes
        self._algorithm = algorithm

        from chszlablib._dyn_matching import DynMatchingSolver
        self._solver = DynMatchingSolver(num_nodes, algorithm, seed)

    def insert_edge(self, u: int, v: int) -> None:
        """Insert an undirected edge (u, v)."""
        self._solver.insert_edge(u, v)

    def delete_edge(self, u: int, v: int) -> None:
        """Delete an undirected edge (u, v)."""
        self._solver.delete_edge(u, v)

    def get_current_solution(self) -> DynMatchingResult:
        """Return the current matching.

        Returns
        -------
        DynMatchingResult
            ``matching_size`` and ``matching`` array
            (matching[v] = mate or -1).
        """
        return DynMatchingResult(
            matching_size=self._solver.get_matching_size(),
            matching=self._solver.get_matching(),
        )


class DynWeightedMIS:
    """Dynamic weighted maximum independent set.

    Maintains a weighted independent set on a dynamic graph where
    edges can be inserted and deleted incrementally. Node weights
    are fixed at construction time.

    Parameters
    ----------
    num_nodes : int
        Number of vertices.
    node_weights : array-like
        Node weight array (length num_nodes, int32).
    algorithm : str, optional
        Algorithm name (default ``"deg_greedy"``).
    seed : int, optional
        Random seed (default 0).
    bfs_depth : int, optional
        BFS depth for local algorithms (default 2).
    time_limit : float, optional
        Time limit for local solver in seconds (default 1.0).
    """

    ALGORITHMS: tuple[str, ...] = (
        "simple", "one_fast", "greedy", "deg_greedy", "bfs",
        "static", "one_strong",
    )
    """Valid algorithm names."""

    DEFAULT_ALGORITHM: str = "deg_greedy"

    def __init__(
        self,
        num_nodes: int,
        node_weights: np.ndarray | list[int],
        algorithm: str = "deg_greedy",
        seed: int = 0,
        bfs_depth: int = 2,
        time_limit: float = 1.0,
    ) -> None:
        if algorithm not in self.ALGORITHMS:
            from chszlablib.exceptions import InvalidModeError
            raise InvalidModeError(
                f"Unknown algorithm {algorithm!r}. Valid: {self.ALGORITHMS}"
            )
        self._num_nodes = num_nodes
        self._algorithm = algorithm

        weights = np.asarray(node_weights, dtype=np.int32)

        from chszlablib._dyn_wmis import DynWMISSolver
        self._solver = DynWMISSolver(
            num_nodes, weights, algorithm, seed, bfs_depth, time_limit,
        )

    def insert_edge(self, u: int, v: int) -> None:
        """Insert an undirected edge (u, v)."""
        self._solver.insert_edge(u, v)

    def delete_edge(self, u: int, v: int) -> None:
        """Delete an undirected edge (u, v)."""
        self._solver.delete_edge(u, v)

    def get_current_solution(self) -> DynWMISResult:
        """Return the current weighted independent set.

        Returns
        -------
        DynWMISResult
            ``weight`` and boolean ``vertices`` array.
        """
        return DynWMISResult(
            weight=self._solver.get_weight(),
            vertices=self._solver.get_mis(),
        )


class DynamicProblems:
    """Dynamic graph algorithms — edge orientation, matching, and weighted MIS.

    Non-instantiable namespace providing factory methods for dynamic solvers.
    Each solver maintains a solution on a graph where edges can be inserted
    and deleted incrementally.
    """

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    @classmethod
    def available_methods(cls) -> dict[str, str]:
        """Return a dict mapping method names to short descriptions."""
        return {
            "edge_orientation": "Dynamic edge orientation — exact (DynDeltaOrientation)",
            "approx_edge_orientation": "Dynamic edge orientation — approximate (DynDeltaApprox)",
            "matching": "Dynamic graph matching (DynMatch)",
            "weighted_mis": "Dynamic weighted MIS (DynWMIS)",
        }

    @staticmethod
    def edge_orientation(
        num_nodes: int,
        algorithm: str = "kflips",
        seed: int = 0,
    ) -> DynEdgeOrientation:
        """Create a dynamic edge orientation solver (exact algorithms).

        Parameters
        ----------
        num_nodes : int
            Number of vertices.
        algorithm : str, optional
            Algorithm name (default ``"kflips"``).
            Valid: ``{DynEdgeOrientation.ALGORITHMS}``.
        seed : int, optional
            Random seed (default 0).

        Returns
        -------
        DynEdgeOrientation
        """
        return DynEdgeOrientation(num_nodes, algorithm=algorithm, seed=seed)

    @staticmethod
    def approx_edge_orientation(
        num_nodes: int,
        num_edges_hint: int = 0,
        algorithm: str = "improved_bfs",
        lambda_param: float = 0.1,
        theta: int = 0,
        b: int = 1,
        bfs_depth: int = 20,
    ) -> DynDeltaApproxOrientation:
        """Create a dynamic edge orientation solver (approximate algorithms).

        Parameters
        ----------
        num_nodes : int
            Number of vertices.
        num_edges_hint : int, optional
            Hint for maximum number of edges (default 0).
        algorithm : str, optional
            Algorithm name (default ``"improved_bfs"``).
            Valid: ``{DynDeltaApproxOrientation.ALGORITHMS}``.
        lambda_param : float, optional
            Lambda parameter for CCHHQRS variants (default 0.1).
        theta : int, optional
            Theta parameter for CCHHQRS variants (default 0).
        b : int, optional
            Fractional edge parameter (default 1).
        bfs_depth : int, optional
            BFS depth for BFS-based algorithms (default 20).

        Returns
        -------
        DynDeltaApproxOrientation
        """
        return DynDeltaApproxOrientation(
            num_nodes, num_edges_hint=num_edges_hint, algorithm=algorithm,
            lambda_param=lambda_param, theta=theta, b=b, bfs_depth=bfs_depth,
        )

    @staticmethod
    def matching(
        num_nodes: int,
        algorithm: str = "blossom",
        seed: int = 0,
    ) -> DynMatching:
        """Create a dynamic matching solver.

        Parameters
        ----------
        num_nodes : int
            Number of vertices.
        algorithm : str, optional
            Algorithm name (default ``"blossom"``).
            Valid: ``{DynMatching.ALGORITHMS}``.
        seed : int, optional
            Random seed (default 0).

        Returns
        -------
        DynMatching
        """
        return DynMatching(num_nodes, algorithm=algorithm, seed=seed)

    @staticmethod
    def weighted_mis(
        num_nodes: int,
        node_weights: np.ndarray | list[int],
        algorithm: str = "deg_greedy",
        seed: int = 0,
        bfs_depth: int = 2,
        time_limit: float = 1.0,
    ) -> DynWeightedMIS:
        """Create a dynamic weighted MIS solver.

        Parameters
        ----------
        num_nodes : int
            Number of vertices.
        node_weights : array-like
            Node weight array (length num_nodes, int32).
        algorithm : str, optional
            Algorithm name (default ``"deg_greedy"``).
            Valid: ``{DynWeightedMIS.ALGORITHMS}``.
        seed : int, optional
            Random seed (default 0).
        bfs_depth : int, optional
            BFS depth for local algorithms (default 2).
        time_limit : float, optional
            Time limit for local solver in seconds (default 1.0).

        Returns
        -------
        DynWeightedMIS
        """
        return DynWeightedMIS(
            num_nodes, node_weights, algorithm=algorithm, seed=seed,
            bfs_depth=bfs_depth, time_limit=time_limit,
        )
