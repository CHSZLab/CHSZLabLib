"""Dynamic edge orientation using DynDeltaOrientation."""
from chszlablib import DynamicProblems

solver = DynamicProblems.edge_orientation(num_nodes=5, algorithm="kflips")
for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
    solver.insert_edge(u, v)

result = solver.get_current_solution()
print(f"Max out-degree: {result.max_out_degree}")
print(f"Out-degrees: {result.out_degrees}")

solver.delete_edge(0, 4)
result = solver.get_current_solution()
print(f"After deleting (0,4) — max out-degree: {result.max_out_degree}")
