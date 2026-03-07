"""Evolutionary graph partitioning using KaFFPaE."""
import sys
from chszlablib import read_metis, Decomposition

# Note for HIGHEST quality use a long time limit
# For even higher quality, use the KaHIP repo implementation that uses MPI and parallism (i.e. all cores of your machine or cluster)
# Python implementation just uses a single core and is not as optimized as the C++ implementation.
g = read_metis(sys.argv[1])
result = Decomposition.evolutionary_partition(g, num_parts=16, time_limit=10, mode="STRONG")
print(f"Edgecut: {result.edgecut}")
print(f"Balance: {result.balance}")
print(f"Assignment: {result.assignment}")
