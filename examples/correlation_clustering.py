"""Correlation clustering on signed graphs using SCC."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.correlation_clustering(g, seed=42)
print(f"Disagreements: {result.edge_cut}")
print(f"Number of clusters: {result.num_clusters}")
print(f"Assignment: {result.assignment}")
