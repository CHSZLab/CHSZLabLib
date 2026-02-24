"""Maximum independent set and maximum weight independent set solvers."""

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


@dataclass
class MWISResult:
    """Result of a maximum weight independent set computation."""

    weight: int
    vertices: np.ndarray


class IndependenceProblems:
    """Maximum independent set and maximum weight independent set solvers."""

    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a namespace and cannot be instantiated")

    # --- KaMIS: Unweighted MIS ---

    @staticmethod
    def redumis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
        full_kernelization: bool = False,
    ) -> MISResult:
        """Compute an unweighted MIS using the ReduMIS evolutionary algorithm."""
        from chszlablib._kamis import redumis as _redumis

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_size, is_verts = _redumis(xadj, adjncy, vwgt, time_limit, seed,
                                     full_kernelization)
        weight = int(np.sum(g.node_weights[is_verts])) if g.node_weights is not None and len(is_verts) > 0 else is_size
        return MISResult(size=is_size, weight=weight, vertices=is_verts)

    @staticmethod
    def online_mis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
        ils_iterations: int = 15000,
    ) -> MISResult:
        """Compute an unweighted MIS using the OnlineMIS local search algorithm."""
        from chszlablib._kamis import online_mis as _online_mis

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_size, is_verts = _online_mis(xadj, adjncy, vwgt, time_limit, seed,
                                        ils_iterations)
        weight = int(np.sum(g.node_weights[is_verts])) if g.node_weights is not None and len(is_verts) > 0 else is_size
        return MISResult(size=is_size, weight=weight, vertices=is_verts)

    # --- KaMIS: Weighted MIS ---

    @staticmethod
    def branch_reduce(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> MISResult:
        """Compute a weighted MIS using the exact branch-and-reduce algorithm."""
        from chszlablib._kamis_wmis import branch_reduce as _branch_reduce

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_weight, is_verts = _branch_reduce(xadj, adjncy, vwgt, time_limit, seed)
        return MISResult(size=len(is_verts), weight=is_weight, vertices=is_verts)

    @staticmethod
    def mmwis(
        g: Graph,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> MISResult:
        """Compute a weighted MIS using the MMWIS memetic evolutionary algorithm."""
        from chszlablib._kamis_mmwis import mmwis_solver as _mmwis_solver

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")

        g.finalize()
        xadj = g.xadj.astype(np.int32, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        vwgt = g.node_weights.astype(np.int32, copy=False) if g.node_weights is not None else np.array([], dtype=np.int32)

        is_weight, is_verts = _mmwis_solver(xadj, adjncy, vwgt, time_limit, seed)
        return MISResult(size=len(is_verts), weight=is_weight, vertices=is_verts)

    # --- CHILS: Weighted MIS ---

    @staticmethod
    def chils(
        g: Graph,
        time_limit: float = 10.0,
        num_concurrent: int = 4,
        seed: int = 0,
    ) -> MWISResult:
        """Compute a maximum weight independent set using CHILS."""
        from chszlablib._chils import mwis as _mwis

        if time_limit < 0:
            raise ValueError(f"time_limit must be >= 0, got {time_limit}")
        if num_concurrent < 1:
            raise ValueError(f"num_concurrent must be >= 1, got {num_concurrent}")

        g.finalize()
        xadj = g.xadj.astype(np.int64, copy=False)
        adjncy = g.adjncy.astype(np.int32, copy=False)
        weights = g.node_weights.astype(np.int64, copy=False)

        total_weight, vertices = _mwis(
            xadj, adjncy, weights, time_limit, num_concurrent, seed,
        )
        return MWISResult(weight=int(total_weight), vertices=vertices)
