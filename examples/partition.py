"""Graph partitioning using KaHIP (KaFFPa)."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.partition(g, num_parts=2, mode="eco")
print(f"Edgecut: {result.edgecut}")
print(f"Assignment: {result.assignment}")
