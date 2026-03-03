"""Demo script: hierarchical process mapping with SharedMap."""

import sys
import time

from chszlablib import Graph, Decomposition


def main():
    if len(sys.argv) < 2:
        # Build a small demo graph if no file provided
        print("No graph file provided, building a small demo graph ...\n")
        g = Graph.from_edge_list([
            (0, 1, 10), (1, 2, 20), (2, 3, 10), (3, 0, 20),
            (0, 2, 5),  (1, 3, 5),
        ])
    else:
        path = sys.argv[1]
        print(f"Loading graph from {path} ...")
        t0 = time.perf_counter()
        g = Graph.from_metis(path)
        dt = time.perf_counter() - t0
        print(f"  Nodes: {g.num_nodes:,}  Edges: {g.num_edges:,}  ({dt:.3f}s)\n")

    # Hierarchy: 2 sockets x 4 cores = 8 PEs total
    hierarchy = [2, 4]
    distance = [10, 1]
    print(f"Hierarchy: {hierarchy}  Distance: {distance}")
    print(f"Total PEs: {hierarchy[0] * hierarchy[1]}\n")

    # Run all modes
    print("=" * 60)
    print("Process Mapping (SharedMap)")
    print("=" * 60)
    for mode in Decomposition.PROCESS_MAP_MODES:
        t0 = time.perf_counter()
        r = Decomposition.process_map(
            g, hierarchy=hierarchy, distance=distance,
            mode=mode, threads=4, seed=42,
        )
        dt = time.perf_counter() - t0
        print(f"  mode={mode:8s}  comm_cost={r.comm_cost:>8,}  ({dt:.3f}s)")
        print(f"    assignment: {r.assignment}")
    print()


if __name__ == "__main__":
    main()
