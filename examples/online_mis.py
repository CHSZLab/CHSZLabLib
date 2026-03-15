"""Maximum independent set using KaMIS (OnlineMIS)."""
import sys
from chszlablib import read_metis, IndependenceProblems

g = read_metis(sys.argv[1])
result = IndependenceProblems.online_mis(g, time_limit=10.0)
print(f"Independent set size: {result.size}")
print(f"Weight: {result.weight}")
print(f"Vertices: {result.vertices}")
