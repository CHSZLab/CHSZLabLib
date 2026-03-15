"""Node separator computation using KaHIP."""
import sys
from chszlablib import read_metis, Decomposition

g = read_metis(sys.argv[1])
result = Decomposition.node_separator(g, mode="eco")
print(f"Separator size: {result.num_separator_vertices}")
print(f"Separator: {result.separator}")
