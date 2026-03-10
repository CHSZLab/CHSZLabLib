"""True streaming hypergraph partitioning using FREIGHT (node-by-node)."""
from chszlablib import FreightPartitioner

# 6 nodes, 3 nets: {0,1,2}, {2,3,4}, {4,5,0}
fp = FreightPartitioner(num_nodes=6, num_nets=3, k=2, seed=42)

blocks = []
blocks.append(fp.assign_node(0, nets=[[0, 1, 2], [0, 4, 5]]))
blocks.append(fp.assign_node(1, nets=[[0, 1, 2]]))
blocks.append(fp.assign_node(2, nets=[[0, 1, 2], [2, 3, 4]]))
blocks.append(fp.assign_node(3, nets=[[2, 3, 4]]))
blocks.append(fp.assign_node(4, nets=[[2, 3, 4], [0, 4, 5]]))
blocks.append(fp.assign_node(5, nets=[[0, 4, 5]]))

print("Immediate block assignments:", blocks)

result = fp.get_assignment()
print("Final assignment array:", result.assignment)
