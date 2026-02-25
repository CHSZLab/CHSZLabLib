"""Maximum cut using fpt-max-cut."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.maxcut(g, method="heuristic", time_limit=1.0)
print(f"Max-cut value: {result.cut_value}")
print(f"Partition: {result.partition}")
