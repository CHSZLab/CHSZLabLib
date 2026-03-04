"""Hypergraph minimum cut using HeiCut."""
from chszlablib import HyperGraph, Decomposition

hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4], [4, 5, 0]])
result = Decomposition.hypergraph_mincut(hg)
print(f"Cut value: {result.cut_value}")
print(f"Time: {result.time:.3f}s")
