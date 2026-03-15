"""Nested dissection ordering using KaHIP."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.node_ordering(g, mode="eco")
print(f"Ordering: {result.ordering}")
