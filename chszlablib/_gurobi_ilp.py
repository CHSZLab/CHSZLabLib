"""Pure Python ILP solver for HyperMIS using gurobipy."""

from __future__ import annotations

import numpy as np


def solve_hypermis_ilp(
    kernel_eptr: np.ndarray,
    kernel_everts: np.ndarray,
    num_nodes: int,
    time_limit: float = 60.0,
) -> tuple[list[int], bool]:
    """Solve the MIS ILP on a reduced hypergraph kernel.

    Formulation: binary variable x[i] per vertex, for each hyperedge
    sum(x[v] for v in edge) <= 1, maximize sum(x).

    Parameters
    ----------
    kernel_eptr : ndarray[int64]
        Edge pointer array (length num_edges + 1).
    kernel_everts : ndarray[int32]
        Concatenated vertex lists per edge.
    num_nodes : int
        Number of vertices in the kernel.
    time_limit : float
        Gurobi time limit in seconds.

    Returns
    -------
    tuple[list[int], bool]
        (selected_vertices, is_optimal) — vertex IDs in the kernel's
        local numbering and whether the solution is provably optimal.
    """
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("hypermis")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # Binary variable per kernel vertex
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")

    # One constraint per hyperedge: at most one vertex selected
    num_edges = len(kernel_eptr) - 1
    for e in range(num_edges):
        start = int(kernel_eptr[e])
        end = int(kernel_eptr[e + 1])
        verts = [int(kernel_everts[i]) for i in range(start, end)]
        if len(verts) >= 2:
            model.addConstr(gp.quicksum(x[v] for v in verts) <= 1)

    model.setObjective(gp.quicksum(x[v] for v in range(num_nodes)), GRB.MAXIMIZE)
    model.optimize()

    selected = [v for v in range(num_nodes) if x[v].X > 0.5]
    is_optimal = model.Status == GRB.OPTIMAL

    return selected, is_optimal
