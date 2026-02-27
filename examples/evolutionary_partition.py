"""Evolutionary graph partitioning using KaFFPaE."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.evolutionary_partition(g, num_parts=16, time_limit=10, mode="FAST")
print(f"Edgecut: {result.edgecut}")
print(f"Balance: {result.balance}")
print(f"Assignment: {result.assignment}")
