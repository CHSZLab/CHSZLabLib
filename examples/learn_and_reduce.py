"""Maximum weight independent set using LearnAndReduce (GNN-guided kernelization)."""
import sys
from chszlablib import read_metis, Graph, IndependenceProblems, LearnAndReduceKernel

# --- Full pipeline (kernelize + solve + lift) ---

if len(sys.argv) > 1:
    g = read_metis(sys.argv[1])
else:
    g = Graph(num_nodes=8)
    for u, v in [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0),
                 (0,3), (1,6)]:
        g.add_edge(u, v)
    for i in range(8):
        g.set_node_weight(i, (i + 1) * 10)

result = IndependenceProblems.learn_and_reduce(
    g, solver="chils", time_limit=5.0, solver_time_limit=2.0,
)
print(f"Full pipeline — weight: {result.weight}, size: {result.size}")
print(f"  vertices: {result.vertices}")

# --- Kernelization-only (two-step workflow) ---

lr = LearnAndReduceKernel(g, config="cyclic_fast", gnn_filter="initial_tight")
kernel = lr.kernelize()
print(f"\nKernelization — kernel nodes: {lr.kernel_nodes}, offset weight: {lr.offset_weight}")

if lr.kernel_nodes > 0:
    kernel_sol = IndependenceProblems.chils(kernel, time_limit=2.0)
    result2 = lr.lift_solution(kernel_sol.vertices)
else:
    import numpy as np
    result2 = lr.lift_solution(np.array([], dtype=np.int32))

print(f"Lifted solution — weight: {result2.weight}, size: {result2.size}")
print(f"  vertices: {result2.vertices}")
