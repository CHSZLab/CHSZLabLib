"""Demo script: load a METIS graph and run all algorithm families."""

import sys
import time

import numpy as np

from chszlablib import (
    Graph, HyperGraph, Decomposition, IndependenceProblems,
    Orientation, DynamicProblems, StreamingBMatcher,
)


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    r = fn(*args, **kwargs)
    return r, time.perf_counter() - t0


def section(num, title):
    print("=" * 60)
    print(f"{num}. {title}")
    print("=" * 60)


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

    # ================================================================
    # DECOMPOSITION
    # ================================================================

    # --- 1. Graph Partitioning (KaHIP) ---
    section(1, "Graph Partitioning (KaHIP)")
    for mode in ["fast", "eco", "strong"]:
        r, dt = timed(Decomposition.partition, g, num_parts=4, mode=mode)
        print(f"  mode={mode:12s}  edgecut={r.edgecut:>8,}  ({dt:.3f}s)")
    print()

    # --- 2. Evolutionary Graph Partitioning (KaFFPaE) ---
    section(2, "Evolutionary Graph Partitioning (KaFFPaE)")
    r, dt = timed(Decomposition.evolutionary_partition, g, num_parts=4, time_limit=5.0, mode="FAST")
    print(f"  edgecut={r.edgecut:>8,}  balance={r.balance:.4f}  ({dt:.3f}s)")
    print()

    # --- 3. Streaming Partitioning (HeiStream) ---
    section(3, "Streaming Partitioning (HeiStream)")
    for mode in ["light", "strong"]:
        r, dt = timed(Decomposition.stream_partition, g, k=4, mode=mode)
        print(f"  mode={mode:8s}  edgecut={r.edgecut:>8,}  ({dt:.3f}s)")
    print()

    # --- 4. Minimum Cut (VieCut) ---
    section(4, "Minimum Cut (VieCut)")
    for algo in ["noi", "viecut", "pr"]:
        r, dt = timed(Decomposition.mincut, g, algorithm=algo)
        side0 = sum(1 for x in r.partition if x == 0)
        side1 = g.num_nodes - side0
        print(f"  algo={algo:8s}  cut_value={r.cut_value:>6,}  "
              f"partition=[{side0:,} | {side1:,}]  ({dt:.3f}s)")
    print()

    # --- 5. Maximum Cut (fpt-max-cut) ---
    section(5, "Maximum Cut (fpt-max-cut)")
    r, dt = timed(Decomposition.maxcut, g, method="heuristic", time_limit=1.0)
    print(f"  cut_value={r.cut_value:>8,}  ({dt:.3f}s)")
    print()

    # --- 6. Graph Clustering (VieClus) ---
    section(6, "Graph Clustering (VieClus)")
    r, dt = timed(Decomposition.cluster, g, time_limit=5.0)
    print(f"  modularity={r.modularity:.6f}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    print()

    # --- 7. Correlation Clustering (SCC) ---
    section(7, "Correlation Clustering (SCC)")
    r, dt = timed(Decomposition.correlation_clustering, g, seed=0)
    print(f"  edge_cut={r.edge_cut:>10,}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    print()

    # --- 8. Evolutionary Correlation Clustering (SCC) ---
    section(8, "Evolutionary Correlation Clustering (SCC)")
    r, dt = timed(Decomposition.evolutionary_correlation_clustering, g, seed=0, time_limit=5.0)
    print(f"  edge_cut={r.edge_cut:>10,}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    print()

    # --- 9. Local Motif Clustering (HeidelbergMotifClustering) ---
    section(9, "Local Motif Clustering (HeidelbergMotifClustering)")
    r, dt = timed(Decomposition.motif_cluster, g, seed_node=0, method="social")
    print(f"  cluster_size={len(r.cluster_nodes):,}  motif_conductance={r.motif_conductance:.4f}  ({dt:.3f}s)")
    print()

    # --- 10. Streaming Clustering (CluStRE) ---
    section(10, "Streaming Clustering (CluStRE)")
    for mode in ["light", "strong"]:
        r, dt = timed(Decomposition.stream_cluster, g, mode=mode)
        print(f"  mode={mode:8s}  clusters={r.num_clusters:,}  modularity={r.modularity:.4f}  ({dt:.3f}s)")
    print()

    # --- 11. Node Separator (KaHIP) ---
    section(11, "Node Separator (KaHIP)")
    r, dt = timed(Decomposition.node_separator, g, mode="eco")
    print(f"  separator_size={r.num_separator_vertices:,}  ({dt:.3f}s)")
    print()

    # --- 12. Node Ordering / Nested Dissection (KaHIP) ---
    section(12, "Node Ordering / Nested Dissection (KaHIP)")
    r, dt = timed(Decomposition.node_ordering, g, mode="eco")
    print(f"  ordering computed, length={len(r.ordering):,}  ({dt:.3f}s)")
    print()

    # --- 13. Process Mapping (SharedMap) ---
    section(13, "Process Mapping (SharedMap)")
    hierarchy = [2, 4]
    distance = [10, 1]
    for mode in Decomposition.PROCESS_MAP_MODES:
        r, dt = timed(Decomposition.process_map, g,
                       hierarchy=hierarchy, distance=distance, mode=mode, threads=4)
        print(f"  mode={mode:8s}  comm_cost={r.comm_cost:>8,}  ({dt:.3f}s)")
    print()

    # ================================================================
    # INDEPENDENCE PROBLEMS
    # ================================================================

    # --- 14. Maximum Weight Independent Set (CHILS) ---
    section(14, "Maximum Weight Independent Set (CHILS)")
    r, dt = timed(IndependenceProblems.chils, g, time_limit=5.0, num_concurrent=4)
    print(f"  weight={r.weight:>12,}  |IS|={len(r.vertices):,}  ({dt:.3f}s)")
    print()

    # --- 15. Maximum Independent Set (KaMIS) ---
    section(15, "Maximum Independent Set (KaMIS)")
    g.finalize()
    for name, fn, kwargs in [
        ("ReduMIS",       IndependenceProblems.redumis,       {"time_limit": 5.0}),
        ("OnlineMIS",     IndependenceProblems.online_mis,    {"time_limit": 5.0, "ils_iterations": 5000}),
        ("Branch&Reduce", IndependenceProblems.branch_reduce, {"time_limit": 10.0}),
        ("MMWIS",         IndependenceProblems.mmwis,         {"time_limit": 5.0}),
    ]:
        r, dt = timed(fn, g, **kwargs)
        print(f"  {name:16s}  |IS|={r.size:>6,}  weight={r.weight:>10,}  ({dt:.3f}s)")
    print()

    # ================================================================
    # ORIENTATION
    # ================================================================

    # --- 16. Edge Orientation (HeiOrient) ---
    section(16, "Edge Orientation (HeiOrient)")
    for algo in ["two_approx", "dfs", "combined"]:
        r, dt = timed(Orientation.orient_edges, g, algorithm=algo)
        print(f"  algo={algo:12s}  max_out_degree={r.max_out_degree:>4}  ({dt:.3f}s)")
    print()

    # ================================================================
    # DYNAMIC PROBLEMS (small demo graph)
    # ================================================================

    # --- 17. Dynamic Edge Orientation (DynDeltaOrientation) ---
    section(17, "Dynamic Edge Orientation (DynDeltaOrientation)")
    solver = DynamicProblems.edge_orientation(num_nodes=5, algorithm="kflips")
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
        solver.insert_edge(u, v)
    r = solver.get_current_solution()
    print(f"  max_out_degree={r.max_out_degree}")
    solver.delete_edge(0, 4)
    r = solver.get_current_solution()
    print(f"  after delete(0,4): max_out_degree={r.max_out_degree}")
    print()

    # --- 18. Dynamic Approx Edge Orientation (DynDeltaApprox) ---
    section(18, "Dynamic Approx Edge Orientation (DynDeltaApprox)")
    solver = DynamicProblems.approx_edge_orientation(num_nodes=5, algorithm="improved_bfs")
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
        solver.insert_edge(u, v)
    max_deg = solver.get_current_solution()
    print(f"  max_out_degree={max_deg}")
    solver.delete_edge(0, 4)
    print(f"  after delete(0,4): max_out_degree={solver.get_current_solution()}")
    print()

    # --- 19. Dynamic Matching (DynMatch) ---
    section(19, "Dynamic Matching (DynMatch)")
    solver = DynamicProblems.matching(num_nodes=6, algorithm="blossom")
    for u, v in [(0, 1), (2, 3), (4, 5), (1, 2)]:
        solver.insert_edge(u, v)
    r = solver.get_current_solution()
    print(f"  matching_size={r.matching_size}")
    solver.delete_edge(0, 1)
    r = solver.get_current_solution()
    print(f"  after delete(0,1): matching_size={r.matching_size}")
    print()

    # --- 20. Dynamic Weighted MIS (DynWMIS) ---
    section(20, "Dynamic Weighted MIS (DynWMIS)")
    weights = np.array([10, 1, 10, 1, 10], dtype=np.int32)
    solver = DynamicProblems.weighted_mis(num_nodes=5, node_weights=weights)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        solver.insert_edge(u, v)
    r = solver.get_current_solution()
    print(f"  MIS weight={r.weight}  vertices={r.vertices}")
    solver.insert_edge(0, 4)
    r = solver.get_current_solution()
    print(f"  after insert(0,4): MIS weight={r.weight}")
    print()

    # ================================================================
    # HYPERGRAPH ALGORITHMS (small demo hypergraph)
    # ================================================================

    hg = HyperGraph.from_edge_list([[0, 1, 2], [2, 3, 4], [4, 5, 0]])

    # --- 21. Hypergraph Minimum Cut (HeiCut) ---
    section(21, "Hypergraph Minimum Cut (HeiCut)")
    r, dt = timed(Decomposition.hypergraph_mincut, hg)
    print(f"  cut_value={r.cut_value}  ({dt:.3f}s)")
    print()

    # --- 22. Hypergraph Independent Set (HyperMIS) ---
    section(22, "Hypergraph Independent Set (HyperMIS)")
    r, dt = timed(IndependenceProblems.hypermis, hg, method="heuristic", time_limit=5.0)
    print(f"  IS size={r.size}  weight={r.weight}  ({dt:.3f}s)")
    print()

    # --- 23. Hypergraph B-Matching (HeiHGM) ---
    section(23, "Hypergraph B-Matching (HeiHGM)")
    r, dt = timed(IndependenceProblems.bmatching, hg, algorithm="greedy_weight_desc")
    print(f"  matched={r.num_matched}  total_weight={r.total_weight}  ({dt:.3f}s)")
    print()

    # --- 24. Streaming Hypergraph Matching (HeiHGM/Streaming) ---
    section(24, "Streaming Hypergraph Matching (HeiHGM/Streaming)")
    sm = StreamingBMatcher(num_nodes=6, algorithm="greedy")
    sm.add_edge([0, 1, 2], weight=1.0)
    sm.add_edge([2, 3, 4], weight=2.0)
    sm.add_edge([4, 5, 0], weight=1.5)
    r = sm.finish()
    print(f"  matched={r.num_matched}  total_weight={r.total_weight}")
    print()

    print("=" * 60)
    print("All 24 algorithm families completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
