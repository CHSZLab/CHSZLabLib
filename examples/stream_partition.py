"""Streaming graph partitioning using HeiStream."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.stream_partition(g, k=2, imbalance=3.0)
print(f"Assignment: {result.assignment}")
