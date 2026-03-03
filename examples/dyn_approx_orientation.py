"""Approximate dynamic edge orientation using DynDeltaApprox."""
from chszlablib import DynamicProblems

solver = DynamicProblems.approx_edge_orientation(num_nodes=5, algorithm="improved_bfs")
for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
    solver.insert_edge(u, v)

max_deg = solver.get_current_solution()
print(f"Max out-degree: {max_deg}")

solver.delete_edge(0, 4)
print(f"After deleting (0,4) — max out-degree: {solver.get_current_solution()}")
