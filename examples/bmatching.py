"""Hypergraph b-matching using HeiHGM."""
from chszlablib import HyperGraph, IndependenceProblems

hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4], [4, 5, 0]])
result = IndependenceProblems.bmatching(hg, algorithm="greedy_weight_desc")
print(f"Matched edges: {result.num_matched}")
print(f"Total weight: {result.total_weight}")
print(f"Matched edge indices: {result.matched_edges}")
