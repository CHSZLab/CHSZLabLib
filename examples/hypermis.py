"""Maximum independent set on hypergraphs using HyperMIS."""
import sys
from chszlablib import read_hmetis, IndependenceProblems

hg = read_hmetis(sys.argv[1])
result = IndependenceProblems.hypermis(hg, method="heuristic", time_limit=60.0)
print(f"Independent set size: {result.size}")
print(f"Weight: {result.weight}")
print(f"Vertices: {result.vertices}")
print(f"Reduction offset: {result.offset}")
print(f"Reduction time: {result.reduction_time:.3f}s")
