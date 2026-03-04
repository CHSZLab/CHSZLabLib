"""Streaming graph clustering using CluStRE."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
for mode in ["light", "strong"]:
    result = Decomposition.stream_cluster(g, mode=mode)
    print(f"Mode: {mode}")
    print(f"  Modularity: {result.modularity:.4f}")
    print(f"  Number of clusters: {result.num_clusters}")
    print(f"  Assignment: {result.assignment}")
