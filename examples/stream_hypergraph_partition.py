"""Streaming hypergraph partitioning using FREIGHT."""
import sys
from chszlablib import read_hmetis, Decomposition

hg = read_hmetis(sys.argv[1])
for algo in ["fennel_approx_sqrt", "fennel", "ldg", "hashing"]:
    result = Decomposition.stream_hypergraph_partition(hg, k=4, algorithm=algo)
    print(f"Algorithm: {algo}")
    print(f"  Assignment: {result.assignment}")
