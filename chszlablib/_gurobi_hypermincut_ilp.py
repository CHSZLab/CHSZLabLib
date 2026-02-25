"""Pure Python ILP solver for hypergraph minimum cut using gurobipy."""

from __future__ import annotations

import time

import numpy as np


def solve_hypermincut_ilp(
    eptr: np.ndarray,
    everts: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
    time_limit: float = 300.0,
) -> tuple[int, float]:
    """Solve hypergraph minimum cut via BIP formulation.

    For each vertex: binary x_v (partition side).
    For each hyperedge e: binary y_e (1 if cut).
    For each e, pick reference v0, then for all v in e:
      y_e >= x_v - x_{v0} and y_e >= x_{v0} - x_v
    Non-trivial: 1 <= sum(x_v) <= n-1
    Minimize: sum(w_e * y_e)
    """
    import gurobipy as gp
    from gurobipy import GRB

    start = time.monotonic()
    num_edges = len(eptr) - 1
    model = gp.Model("hypermincut")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_edges, vtype=GRB.BINARY, name="y")

    for e in range(num_edges):
        s, t = int(eptr[e]), int(eptr[e + 1])
        verts = [int(everts[i]) for i in range(s, t)]
        if len(verts) < 2:
            continue
        v0 = verts[0]
        for v in verts[1:]:
            model.addConstr(y[e] >= x[v] - x[v0])
            model.addConstr(y[e] >= x[v0] - x[v])

    model.addConstr(gp.quicksum(x[v] for v in range(num_nodes)) >= 1)
    model.addConstr(gp.quicksum(x[v] for v in range(num_nodes)) <= num_nodes - 1)

    model.setObjective(
        gp.quicksum(int(edge_weights[e]) * y[e] for e in range(num_edges)),
        GRB.MINIMIZE,
    )
    model.optimize()

    cut_value = int(round(model.ObjVal))
    elapsed = time.monotonic() - start
    return cut_value, elapsed
