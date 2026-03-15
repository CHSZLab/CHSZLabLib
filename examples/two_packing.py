"""Maximum 2-packing set using red2pack."""
import sys
from chszlablib import read_metis, IndependenceProblems

g = read_metis(sys.argv[1])
result = IndependenceProblems.two_packing(g, algorithm="chils", time_limit=10.0)
print(f"2-packing size: {result.size}")
print(f"Weight: {result.weight}")
print(f"Vertices: {result.vertices}")
