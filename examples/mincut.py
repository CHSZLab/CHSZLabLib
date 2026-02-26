"""Global minimum cut using VieCut."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.mincut(g, algorithm="inexact")
print(f"Min-cut value: {result.cut_value}")
print(f"Partition: {result.partition}")
