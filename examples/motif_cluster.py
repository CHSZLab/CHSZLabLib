"""Local motif clustering using HeidelbergMotifClustering."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.motif_cluster(g, seed_node=0, method="social")
print(f"Cluster size: {len(result.cluster_nodes)}")
print(f"Motif conductance: {result.motif_conductance:.4f}")
print(f"Cluster nodes: {result.cluster_nodes}")
