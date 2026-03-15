"""Community detection / modularity maximization using VieClus."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.cluster(g, time_limit=1.0)
print(f"Modularity: {result.modularity:.4f}")
print(f"Number of clusters: {result.num_clusters}")
print(f"Assignment: {result.assignment}")
