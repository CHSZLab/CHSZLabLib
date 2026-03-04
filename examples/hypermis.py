"""Maximum independent set on hypergraphs using HyperMIS."""
import sys
from chszlablib import HyperGraph, IndependenceProblems

if len(sys.argv) > 1:
    from chszlablib import read_hmetis
    hg = read_hmetis(sys.argv[1])
else:
    hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4], [4, 5, 0]])

result = IndependenceProblems.hypermis(hg, method="heuristic", time_limit=60.0)
print(f"Independent set size: {result.size}")
print(f"Weight: {result.weight}")
print(f"Vertices: {result.vertices}")
print(f"Reduction offset: {result.offset}")
print(f"Reduction time: {result.reduction_time:.3f}s")
