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


def run_section(num, title, fn, counts):
    """Run a demo section, handling missing optional modules gracefully."""
    section(num, title)
    try:
        fn()
        counts["ok"] += 1
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [SKIPPED] {e}")
        counts["skip"] += 1
    print()


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

    counts = {"ok": 0, "skip": 0}

    # ================================================================
    # DECOMPOSITION
    # ================================================================

    # --- 1. Graph Partitioning (KaHIP) ---
    def s1():
        for mode in ["fast", "eco", "strong"]:
            r, dt = timed(Decomposition.partition, g, num_parts=4, mode=mode)
            print(f"  mode={mode:12s}  edgecut={r.edgecut:>8,}  ({dt:.3f}s)")
    run_section(1, "Graph Partitioning (KaHIP)", s1, counts)

    # --- 2. Evolutionary Graph Partitioning (KaFFPaE) ---
    def s2():
        r, dt = timed(Decomposition.evolutionary_partition, g, num_parts=4, time_limit=5, mode="FAST")
        print(f"  edgecut={r.edgecut:>8,}  balance={r.balance:.4f}  ({dt:.3f}s)")
    run_section(2, "Evolutionary Graph Partitioning (KaFFPaE)", s2, counts)

    # --- 3. Streaming Partitioning (HeiStream) ---
    def s3():
        r, dt = timed(Decomposition.stream_partition, g, k=4)
        print(f"  fennel          parts={len(np.unique(r.assignment)):>2}  ({dt:.3f}s)")
        r, dt = timed(Decomposition.stream_partition, g, k=4, max_buffer_size=1000, num_streams_passes=2)
        print(f"  buffcut(2pass)  parts={len(np.unique(r.assignment)):>2}  ({dt:.3f}s)")
    run_section(3, "Streaming Partitioning (HeiStream)", s3, counts)

    # --- 4. Minimum Cut (VieCut) ---
    def s4():
        for algo in ["cactus", "exact", "inexact"]:
            r, dt = timed(Decomposition.mincut, g, algorithm=algo)
            side0 = sum(1 for x in r.partition if x == 0)
            side1 = g.num_nodes - side0
            print(f"  algo={algo:8s}  cut_value={r.cut_value:>6,}  "
                  f"partition=[{side0:,} | {side1:,}]  ({dt:.3f}s)")
    run_section(4, "Minimum Cut (VieCut)", s4, counts)

    # --- 5. Maximum Cut (fpt-max-cut) ---
    def s5():
        r, dt = timed(Decomposition.maxcut, g, method="heuristic", time_limit=1.0)
        print(f"  cut_value={r.cut_value:>8,}  ({dt:.3f}s)")
    run_section(5, "Maximum Cut (fpt-max-cut)", s5, counts)

    # --- 6. Graph Clustering (VieClus) ---
    def s6():
        r, dt = timed(Decomposition.cluster, g, time_limit=5.0)
        print(f"  modularity={r.modularity:.6f}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    run_section(6, "Graph Clustering (VieClus)", s6, counts)

    # --- 7. Correlation Clustering (SCC) ---
    def s7():
        r, dt = timed(Decomposition.correlation_clustering, g, seed=0)
        print(f"  edge_cut={r.edge_cut:>10,}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    run_section(7, "Correlation Clustering (SCC)", s7, counts)

    # --- 8. Evolutionary Correlation Clustering (SCC) ---
    def s8():
        r, dt = timed(Decomposition.evolutionary_correlation_clustering, g, seed=0, time_limit=5.0)
        print(f"  edge_cut={r.edge_cut:>10,}  clusters={r.num_clusters:,}  ({dt:.3f}s)")
    run_section(8, "Evolutionary Correlation Clustering (SCC)", s8, counts)

    # --- 9. Local Motif Clustering (HeidelbergMotifClustering) ---
    def s9():
        r, dt = timed(Decomposition.motif_cluster, g, seed_node=0, method="social")
        print(f"  cluster_size={len(r.cluster_nodes):,}  motif_conductance={r.motif_conductance:.4f}  ({dt:.3f}s)")
    run_section(9, "Local Motif Clustering (HeidelbergMotifClustering)", s9, counts)

    # --- 10. Streaming Clustering (CluStRE) ---
    def s10():
        for mode in ["light", "strong"]:
            r, dt = timed(Decomposition.stream_cluster, g, mode=mode)
            print(f"  mode={mode:8s}  clusters={r.num_clusters:,}  modularity={r.modularity:.4f}  ({dt:.3f}s)")
    run_section(10, "Streaming Clustering (CluStRE)", s10, counts)

    # --- 11. Node Separator (KaHIP) ---
    def s11():
        r, dt = timed(Decomposition.node_separator, g, mode="eco")
        print(f"  separator_size={r.num_separator_vertices:,}  ({dt:.3f}s)")
    run_section(11, "Node Separator (KaHIP)", s11, counts)

    # --- 12. Node Ordering / Nested Dissection (KaHIP) ---
    def s12():
        r, dt = timed(Decomposition.node_ordering, g, mode="eco")
        print(f"  ordering computed, length={len(r.ordering):,}  ({dt:.3f}s)")
    run_section(12, "Node Ordering / Nested Dissection (KaHIP)", s12, counts)

    # --- 13. Process Mapping (SharedMap) ---
    def s13():
        hierarchy = [2, 4]
        distance = [10, 1]
        for mode in Decomposition.PROCESS_MAP_MODES:
            r, dt = timed(Decomposition.process_map, g,
                           hierarchy=hierarchy, distance=distance, mode=mode, threads=4)
            print(f"  mode={mode:8s}  comm_cost={r.comm_cost:>8,}  ({dt:.3f}s)")
    run_section(13, "Process Mapping (SharedMap)", s13, counts)

    # ================================================================
    # INDEPENDENCE PROBLEMS
    # ================================================================

    # --- 14. Maximum Weight Independent Set (CHILS) ---
    def s14():
        r, dt = timed(IndependenceProblems.chils, g, time_limit=5.0, num_concurrent=4)
        print(f"  weight={r.weight:>12,}  |IS|={len(r.vertices):,}  ({dt:.3f}s)")
    run_section(14, "Maximum Weight Independent Set (CHILS)", s14, counts)

    # --- 15. Maximum Independent Set (KaMIS) ---
    def s15():
        g.finalize()
        for name, fn, kwargs in [
            ("ReduMIS",       IndependenceProblems.redumis,       {"time_limit": 5.0}),
            ("OnlineMIS",     IndependenceProblems.online_mis,    {"time_limit": 5.0, "ils_iterations": 5000}),
            ("Branch&Reduce", IndependenceProblems.branch_reduce, {"time_limit": 10.0}),
            ("MMWIS",         IndependenceProblems.mmwis,         {"time_limit": 5.0}),
        ]:
            r, dt = timed(fn, g, **kwargs)
            print(f"  {name:16s}  |IS|={r.size:>6,}  weight={r.weight:>10,}  ({dt:.3f}s)")
    run_section(15, "Maximum Independent Set (KaMIS)", s15, counts)

    # ================================================================
    # ORIENTATION
    # ================================================================

    # --- 16. Edge Orientation (HeiOrient) ---
    def s16():
        for algo in ["two_approx", "dfs", "combined"]:
            r, dt = timed(Orientation.orient_edges, g, algorithm=algo)
            print(f"  algo={algo:12s}  max_out_degree={r.max_out_degree:>4}  ({dt:.3f}s)")
    run_section(16, "Edge Orientation (HeiOrient)", s16, counts)

    # ================================================================
    # DYNAMIC PROBLEMS (small demo graph)
    # ================================================================

    # --- 17. Dynamic Edge Orientation (DynDeltaOrientation) ---
    def s17():
        solver = DynamicProblems.edge_orientation(num_nodes=5, algorithm="kflips")
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
            solver.insert_edge(u, v)
        r = solver.get_current_solution()
        print(f"  max_out_degree={r.max_out_degree}")
        solver.delete_edge(0, 4)
        r = solver.get_current_solution()
        print(f"  after delete(0,4): max_out_degree={r.max_out_degree}")
    run_section(17, "Dynamic Edge Orientation (DynDeltaOrientation)", s17, counts)

    # --- 18. Dynamic Approx Edge Orientation (DynDeltaApprox) ---
    def s18():
        solver = DynamicProblems.approx_edge_orientation(num_nodes=5, algorithm="improved_bfs")
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
            solver.insert_edge(u, v)
        max_deg = solver.get_current_solution()
        print(f"  max_out_degree={max_deg}")
        solver.delete_edge(0, 4)
        print(f"  after delete(0,4): max_out_degree={solver.get_current_solution()}")
    run_section(18, "Dynamic Approx Edge Orientation (DynDeltaApprox)", s18, counts)

    # --- 19. Dynamic Matching (DynMatch) ---
    def s19():
        solver = DynamicProblems.matching(num_nodes=6, algorithm="blossom")
        for u, v in [(0, 1), (2, 3), (4, 5), (1, 2)]:
            solver.insert_edge(u, v)
        r = solver.get_current_solution()
        print(f"  matching_size={r.matching_size}")
        solver.delete_edge(0, 1)
        r = solver.get_current_solution()
        print(f"  after delete(0,1): matching_size={r.matching_size}")
    run_section(19, "Dynamic Matching (DynMatch)", s19, counts)

    # --- 20. Dynamic Weighted MIS (DynWMIS) ---
    def s20():
        weights = np.array([10, 1, 10, 1, 10], dtype=np.int32)
        solver = DynamicProblems.weighted_mis(num_nodes=5, node_weights=weights)
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            solver.insert_edge(u, v)
        r = solver.get_current_solution()
        print(f"  MIS weight={r.weight}  vertices={r.vertices}")
        solver.insert_edge(0, 4)
        r = solver.get_current_solution()
        print(f"  after insert(0,4): MIS weight={r.weight}")
    run_section(20, "Dynamic Weighted MIS (DynWMIS)", s20, counts)

    # ================================================================
    # HYPERGRAPH ALGORITHMS (convert input graph to hypergraph)
    # ================================================================

    print("Converting graph to hypergraph (each edge -> size-2 hyperedge) ...")
    t0 = time.perf_counter()
    g.finalize()
    edges = []
    for u in range(g.num_nodes):
        for idx in range(g.xadj[u], g.xadj[u + 1]):
            v = int(g.adjncy[idx])
            if u < v:
                edges.append([u, v])
    hg = HyperGraph.from_edge_list(edges, num_nodes=g.num_nodes)
    dt_conv = time.perf_counter() - t0
    print(f"  Nodes: {hg.num_nodes:,}  Hyperedges: {hg.num_edges:,}  ({dt_conv:.3f}s)\n")

    # --- 21. Hypergraph Minimum Cut (HeiCut) ---
    def s21():
        r, dt = timed(Decomposition.hypergraph_mincut, hg)
        print(f"  cut_value={r.cut_value}  ({dt:.3f}s)")
    run_section(21, "Hypergraph Minimum Cut (HeiCut)", s21, counts)

    # --- 22. Hypergraph Independent Set (HyperMIS) ---
    def s22():
        r, dt = timed(IndependenceProblems.hypermis, hg, method="heuristic", time_limit=5.0)
        print(f"  IS size={r.size}  weight={r.weight}  ({dt:.3f}s)")
    run_section(22, "Hypergraph Independent Set (HyperMIS)", s22, counts)

    # --- 23. Hypergraph B-Matching (HeiHGM) ---
    def s23():
        r, dt = timed(IndependenceProblems.bmatching, hg, algorithm="greedy_weight_desc")
        print(f"  matched={r.num_matched}  total_weight={r.total_weight}  ({dt:.3f}s)")
    run_section(23, "Hypergraph B-Matching (HeiHGM)", s23, counts)

    # --- 24. Streaming Hypergraph Matching (HeiHGM/Streaming) ---
    def s24():
        g.finalize()
        sm = StreamingBMatcher(num_nodes=g.num_nodes, algorithm="greedy")
        for u in range(g.num_nodes):
            for idx in range(g.xadj[u], g.xadj[u + 1]):
                v = int(g.adjncy[idx])
                if u < v:
                    sm.add_edge([u, v], weight=float(g.edge_weights[idx]))
        r = sm.finish()
        print(f"  matched={r.num_matched}  total_weight={r.total_weight}")
    run_section(24, "Streaming Hypergraph Matching (HeiHGM/Streaming)", s24, counts)

    # ================================================================
    print("=" * 60)
    total = counts["ok"] + counts["skip"]
    print(f"Done: {counts['ok']}/{total} passed, {counts['skip']} skipped (optional modules not installed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
