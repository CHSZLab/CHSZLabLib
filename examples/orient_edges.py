"""Edge orientation to minimize maximum out-degree using HeiOrient."""
import sys
from chszlablib import read_metis, Orientation

g = read_metis(sys.argv[1])
result = Orientation.orient_edges(g, algorithm="combined")
print(f"Max out-degree: {result.max_out_degree}")
print(f"Out-degrees: {result.out_degrees}")
print(f"Edge heads: {result.edge_heads}")
