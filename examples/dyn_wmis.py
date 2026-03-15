"""Dynamic weighted maximum independent set using DynWMIS."""
import numpy as np
from chszlablib import DynamicProblems

weights = np.array([10, 1, 10, 1, 10], dtype=np.int32)
solver = DynamicProblems.weighted_mis(num_nodes=5, node_weights=weights)
for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
    solver.insert_edge(u, v)

result = solver.get_current_solution()
print(f"MIS weight: {result.weight}")
print(f"In MIS: {result.vertices}")

solver.insert_edge(0, 4)
result = solver.get_current_solution()
print(f"After adding (0,4) — MIS weight: {result.weight}")
