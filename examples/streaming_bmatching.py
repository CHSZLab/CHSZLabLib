"""Streaming hypergraph b-matching using HeiHGM/Streaming."""
from chszlablib import StreamingBMatcher

sm = StreamingBMatcher(num_nodes=6, algorithm="greedy")
sm.add_edge([0, 1, 2], weight=1.0)
sm.add_edge([2, 3, 4], weight=2.0)
sm.add_edge([4, 5, 0], weight=1.5)

result = sm.finish()
print(f"Matched edges: {result.num_matched}")
print(f"Total weight: {result.total_weight}")
print(f"Matched edge indices: {result.matched_edges}")
