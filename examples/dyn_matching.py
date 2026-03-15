"""Dynamic graph matching using DynMatch."""
from chszlablib import DynamicProblems

solver = DynamicProblems.matching(num_nodes=6, algorithm="blossom")
for u, v in [(0, 1), (2, 3), (4, 5), (1, 2)]:
    solver.insert_edge(u, v)

result = solver.get_current_solution()
print(f"Matching size: {result.matching_size}")
print(f"Matching: {result.matching}")

solver.delete_edge(0, 1)
result = solver.get_current_solution()
print(f"After deleting (0,1) — matching size: {result.matching_size}")
