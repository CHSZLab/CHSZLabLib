"""Demo script: load a METIS graph and run all algorithm families."""

import sys
import time

from chszlablib import (
    Graph, partition, mincut, cluster, mwis,
    redumis, online_mis, branch_reduce, mmwis_solver,
    correlation_clustering,
)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <metis_graph_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading graph from {path} ...")
    t0 = time.perf_counter()
    g = Graph.from_metis(path)
    t_load = time.perf_counter() - t0
    print(f"  Nodes: {g.num_nodes:,}  Edges: {g.num_edges:,}  ({t_load:.3f}s)\n")

    # --- 1. Graph Partitioning (KaHIP) ---
    print("=" * 60)
    print("1. Graph Partitioning (KaHIP)")
    print("=" * 60)
    for mode in ["fast", "eco", "strong"]:
        t0 = time.perf_counter()
        r = partition(g, num_parts=4, mode=mode)
        dt = time.perf_counter() - t0
        print(f"  mode={mode:12s}  edgecut={r.edgecut:>8,}  ({dt:.3f}s)")
    print()

    # --- 2. Minimum Cut (VieCut) ---
    print("=" * 60)
    print("2. Minimum Cut (VieCut)")
    print("=" * 60)
    for algo in ["noi", "viecut", "pr"]:
        t0 = time.perf_counter()
        r = mincut(g, algorithm=algo)
        dt = time.perf_counter() - t0
        side0 = sum(1 for x in r.partition if x == 0)
        side1 = g.num_nodes - side0
        print(f"  algo={algo:8s}  cut_value={r.cut_value:>6,}  "
              f"partition=[{side0:,} | {side1:,}]  ({dt:.3f}s)")
    print()

    # --- 3. Graph Clustering (VieClus) ---
    print("=" * 60)
    print("3. Graph Clustering (VieClus)")
    print("=" * 60)
    t0 = time.perf_counter()
    r = cluster(g, time_limit=5.0)
    dt = time.perf_counter() - t0
    print(f"  modularity={r.modularity:.6f}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    print()

    # --- 4. Maximum Weight Independent Set (CHILS) ---
    print("=" * 60)
    print("4. Maximum Weight Independent Set (CHILS)")
    print("=" * 60)
    t0 = time.perf_counter()
    r = mwis(g, time_limit=5.0, num_concurrent=4)
    dt = time.perf_counter() - t0
    print(f"  weight={r.weight:>12,}  |IS|={len(r.vertices):,}  ({dt:.3f}s)")

    # Quick validity check
    g.finalize()
    is_set = set(r.vertices)
    valid = True
    for u in is_set:
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            if g.adjncy[idx] in is_set:
                valid = False
                break
        if not valid:
            break
    print(f"  independent set valid: {valid}")
    print()

    # --- 5. Maximum Independent Set (KaMIS) ---
    print("=" * 60)
    print("5. Maximum Independent Set (KaMIS)")
    print("=" * 60)

    for name, fn, kwargs in [
        ("ReduMIS",         redumis,        {"time_limit": 5.0}),
        ("OnlineMIS",       online_mis,     {"time_limit": 5.0, "ils_iterations": 5000}),
        ("Branch&Reduce",   branch_reduce,  {"time_limit": 10.0}),
        ("MMWIS",           mmwis_solver,   {"time_limit": 5.0}),
    ]:
        t0 = time.perf_counter()
        r = fn(g, **kwargs)
        dt = time.perf_counter() - t0
        # Verify IS validity
        is_set = set(r.vertices)
        valid = True
        for u in is_set:
            for idx in range(g.xadj[u], g.xadj[u + 1]):
                if g.adjncy[idx] in is_set:
                    valid = False
                    break
            if not valid:
                break
        print(f"  {name:16s}  |IS|={r.size:>6,}  weight={r.weight:>10,}  "
              f"valid={valid}  ({dt:.3f}s)")
    print()

    # --- 6. Correlation Clustering (SCC) ---
    print("=" * 60)
    print("6. Correlation Clustering (SCC)")
    print("=" * 60)
    # Note: correlation clustering expects signed edge weights.
    # On a standard (unsigned) graph, all edges are treated as positive.
    t0 = time.perf_counter()
    r = correlation_clustering(g, seed=0)
    dt = time.perf_counter() - t0
    print(f"  edge_cut={r.edge_cut:>10,}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    print()

    print("All algorithms completed successfully.")


if __name__ == "__main__":
    main()
