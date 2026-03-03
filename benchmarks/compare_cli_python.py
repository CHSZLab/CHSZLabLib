"""Compare CLI and Python interface for dynamic algorithms and SharedMap.

Uses 10th DIMACS challenge graphs (delaunay_n15, rgg_n_2_15_s0, astro-ph, as-22july06).
Reports quality and timing differences.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTREP = os.path.join(ROOT, "external_repositories")

ORIENTATION_CLI = os.path.join(EXTREP, "DynDeltaOrientation/build/delta-orientations")
DYNMATCH_CLI = os.path.join(EXTREP, "DynMatch/deploy/dynmatch")
DYNWMIS_CLI = os.path.join(EXTREP, "DynWMIS/deploy/dynwmis")
SHAREDMAP_CLI = os.path.join(EXTREP, "SharedMap/build/SharedMap")

MTKAHYPAR_LIB_DIR = os.path.join(EXTREP, "HeiCut/extern/mt-kahypar-library")

# Graphs (.seq files for dynamic, .graph files for SharedMap)
GRAPHS_SEQ = {
    "delaunay_n15": os.path.join(EXTREP, "KaHIP/examples/delaunay_n15.graph.seq"),
    "as-22july06": os.path.join(EXTREP, "VieClus/examples/as-22july06.graph.seq"),
}
GRAPHS_METIS = {
    "delaunay_n15": os.path.join(EXTREP, "KaHIP/examples/delaunay_n15.graph"),
    "as-22july06": os.path.join(EXTREP, "VieClus/examples/as-22july06.graph"),
}

# Small DynMatch seq with deletions
DYNMATCH_SEQ = {
    "munmun_digg": os.path.join(EXTREP, "DynMatch/examples/munmun_digg.undo.0.1.seq"),
    "wordassoc": os.path.join(EXTREP, "DynMatch/examples/wordassociation-2011.graph.seq"),
}

# DynWMIS seq (unweighted)
DYNWMIS_SEQ = {
    "3elt": os.path.join(EXTREP, "DynWMIS/examples/3elt.graph.seq"),
    "4elt": os.path.join(EXTREP, "DynWMIS/examples/4elt.graph.seq"),
}


def parse_seq_file(path):
    """Parse a .seq file into (n, edge_operations) list."""
    ops = []
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split()
                n = int(parts[0])
                continue
            parts = line.split()
            op = int(parts[0])  # 1=insert, 0=delete
            u, v = int(parts[1]), int(parts[2])
            ops.append((op, u, v))
    return n, ops


def run_cli(cmd, env=None):
    """Run CLI command, return (stdout, elapsed_seconds)."""
    my_env = os.environ.copy()
    if env:
        my_env.update(env)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=my_env, timeout=300)
    elapsed = time.perf_counter() - t0
    return result.stdout + result.stderr, elapsed


# ===========================================================================
# DynDeltaOrientation
# ===========================================================================
def benchmark_dyn_orientation():
    print("\n" + "=" * 70)
    print("DynDeltaOrientation: Python vs CLI")
    print("=" * 70)

    from chszlablib.dynamic import DynEdgeOrientation

    # Algorithm name mapping: Python -> CLI
    algo_map = {
        "kflips": "KFLIPS",
        "bfs": "BFSCS",
        "naive": "NAIVE",
        "improved_opt": "IMPROVEDOPT",
    }

    for graph_name, seq_path in GRAPHS_SEQ.items():
        if not os.path.exists(seq_path):
            print(f"  SKIP {graph_name}: {seq_path} not found")
            continue
        n, ops = parse_seq_file(seq_path)
        print(f"\n  Graph: {graph_name} (n={n}, ops={len(ops)})")

        for py_algo, cli_algo in algo_map.items():
            seed = 42

            # --- CLI ---
            if os.path.exists(ORIENTATION_CLI):
                out, cli_time = run_cli([
                    ORIENTATION_CLI, seq_path,
                    f"--algorithm={cli_algo}", f"--seed={seed}",
                ])
                cli_maxdeg = None
                cli_algo_time = None
                for line in out.split("\n"):
                    if "maxOutDegree" in line:
                        cli_maxdeg = int(line.split()[-1])
                    if line.startswith("time"):
                        cli_algo_time = float(line.split()[-1])
            else:
                cli_maxdeg = None
                cli_algo_time = None

            # --- Python ---
            solver = DynEdgeOrientation(num_nodes=n, algorithm=py_algo, seed=seed)
            t0 = time.perf_counter()
            for op, u, v in ops:
                if op == 1:
                    solver.insert_edge(u, v)
                else:
                    solver.delete_edge(u, v)
            py_time = time.perf_counter() - t0
            py_result = solver.get_current_solution()
            py_maxdeg = py_result.max_out_degree

            match_str = ""
            if cli_maxdeg is not None:
                if py_maxdeg == cli_maxdeg:
                    match_str = "MATCH"
                else:
                    match_str = f"MISMATCH (cli={cli_maxdeg})"

            cli_time_str = f"{cli_algo_time:.3f}s" if cli_algo_time else "N/A"
            print(f"    {py_algo:20s} maxdeg={py_maxdeg:4d}  py={py_time:.3f}s  cli={cli_time_str}  {match_str}")


# ===========================================================================
# DynMatch
# ===========================================================================
def benchmark_dyn_matching():
    print("\n" + "=" * 70)
    print("DynMatching: Python vs CLI")
    print("=" * 70)

    from chszlablib.dynamic import DynMatching

    algo_map = {
        "random_walk": "randomwalk",
        "blossom": "dynblossom",
        "naive": "naive",
        "baswana_gupta_sen": "baswanaguptasen",
        "neiman_solomon": "neimansolomon",
    }

    all_seqs = {**GRAPHS_SEQ, **DYNMATCH_SEQ}
    for graph_name, seq_path in all_seqs.items():
        if not os.path.exists(seq_path):
            print(f"  SKIP {graph_name}: {seq_path} not found")
            continue
        n, ops = parse_seq_file(seq_path)
        print(f"\n  Graph: {graph_name} (n={n}, ops={len(ops)})")

        for py_algo, cli_algo in algo_map.items():
            seed = 42

            # --- CLI ---
            if os.path.exists(DYNMATCH_CLI):
                out, cli_time = run_cli([
                    DYNMATCH_CLI, seq_path,
                    f"--algorithm={cli_algo}", f"--seed={seed}",
                ])
                # CLI prints lines like: "matching_size  elapsed_time"
                cli_msize = None
                cli_algo_time = None
                for line in out.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("io "):
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            cli_msize = int(parts[0])
                            cli_algo_time = float(parts[1])
                        except ValueError:
                            pass
            else:
                cli_msize = None
                cli_algo_time = None

            # --- Python ---
            solver = DynMatching(num_nodes=n, algorithm=py_algo, seed=seed)
            t0 = time.perf_counter()
            for op, u, v in ops:
                if op == 1:
                    solver.insert_edge(u, v)
                else:
                    solver.delete_edge(u, v)
            py_time = time.perf_counter() - t0
            py_result = solver.get_current_solution()
            py_msize = py_result.matching_size

            match_str = ""
            if cli_msize is not None:
                if py_msize == cli_msize:
                    match_str = "MATCH"
                else:
                    diff_pct = abs(py_msize - cli_msize) / max(cli_msize, 1) * 100
                    match_str = f"DIFF cli={cli_msize} ({diff_pct:+.1f}%)"

            cli_time_str = f"{cli_algo_time:.3f}s" if cli_algo_time else "N/A"
            print(f"    {py_algo:20s} msize={py_msize:6d}  py={py_time:.3f}s  cli={cli_time_str}  {match_str}")


# ===========================================================================
# DynWMIS
# ===========================================================================
def benchmark_dyn_wmis():
    print("\n" + "=" * 70)
    print("DynWMIS: Python vs CLI")
    print("=" * 70)

    from chszlablib.dynamic import DynWeightedMIS

    algo_map = {
        "deg_greedy": "DegGreedy",
        "one_fast": "DynamicOneFast",
        "one_strong": "DynamicOneStrong",
    }

    for graph_name, seq_path in DYNWMIS_SEQ.items():
        if not os.path.exists(seq_path):
            print(f"  SKIP {graph_name}: {seq_path} not found")
            continue
        n, ops = parse_seq_file(seq_path)
        print(f"\n  Graph: {graph_name} (n={n}, ops={len(ops)})")

        for py_algo, cli_algo in algo_map.items():
            seed = 42

            # --- CLI ---
            if os.path.exists(DYNWMIS_CLI):
                out, cli_time = run_cli([
                    DYNWMIS_CLI, seq_path,
                    f"--algorithm={cli_algo}", f"--seed={seed}",
                ])
                cli_weight = None
                cli_algo_time = None
                for line in out.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("weight"):
                        cli_weight = int(line.split()[-1])
                    if line.startswith("time"):
                        cli_algo_time = float(line.split()[-1])
                    if line.startswith("mis size checked"):
                        cli_weight = int(line.split()[-1])
            else:
                cli_weight = None
                cli_algo_time = None

            # --- Python ---
            # Unweighted: all weights = 1
            weights = np.ones(n, dtype=np.int32)
            solver = DynWeightedMIS(
                num_nodes=n, node_weights=weights,
                algorithm=py_algo, seed=seed,
                bfs_depth=10, time_limit=1000.0,
            )
            t0 = time.perf_counter()
            for op, u, v in ops:
                if op == 1:
                    solver.insert_edge(u, v)
                else:
                    solver.delete_edge(u, v)
            py_time = time.perf_counter() - t0
            py_result = solver.get_current_solution()
            py_weight = py_result.weight

            match_str = ""
            if cli_weight is not None:
                if py_weight == cli_weight:
                    match_str = "MATCH"
                else:
                    diff_pct = (py_weight - cli_weight) / max(cli_weight, 1) * 100
                    match_str = f"DIFF cli={cli_weight} ({diff_pct:+.1f}%)"

            cli_time_str = f"{cli_algo_time:.3f}s" if cli_algo_time else "N/A"
            print(f"    {py_algo:20s} weight={py_weight:6d}  py={py_time:.3f}s  cli={cli_time_str}  {match_str}")


# ===========================================================================
# SharedMap
# ===========================================================================
def benchmark_sharedmap():
    print("\n" + "=" * 70)
    print("SharedMap: Python vs CLI")
    print("=" * 70)

    from chszlablib import Graph, Decomposition

    configs = ["fast", "eco", "strong"]
    hierarchy = [4, 4]
    distance = [10, 1]
    hier_str = ":".join(str(h) for h in hierarchy)
    dist_str = ":".join(str(d) for d in distance)
    # Use 1 thread for reproducible comparison (TBB is non-deterministic with >1 thread)
    threads = 1

    sharedmap_lib_dir = os.path.join(EXTREP, "SharedMap/extern/local/mt-kahypar/lib")
    # Try lib64 too
    if not os.path.isdir(sharedmap_lib_dir):
        sharedmap_lib_dir = os.path.join(EXTREP, "SharedMap/extern/local/mt-kahypar/lib64")

    for graph_name, metis_path in GRAPHS_METIS.items():
        if not os.path.exists(metis_path):
            print(f"  SKIP {graph_name}: {metis_path} not found")
            continue

        g = Graph.from_metis(metis_path)
        print(f"\n  Graph: {graph_name} (n={g.num_nodes}, m={g.num_edges})")

        for config in configs:
            seed = 42

            # --- CLI ---
            if os.path.exists(SHAREDMAP_CLI):
                ld_path = sharedmap_lib_dir
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".map", delete=True) as tmp:
                    tmp_path = tmp.name
                out, cli_time = run_cli(
                    [SHAREDMAP_CLI,
                     "-g", metis_path,
                     "-h", hier_str,
                     "-d", dist_str,
                     "-c", config,
                     "-t", str(threads),
                     "--seed", str(seed),
                     "-e", "0.03",
                     "-m", tmp_path,
                     "-s", "nb_layer"],
                    env={"LD_LIBRARY_PATH": ld_path},
                )
                cli_cost = None
                for line in out.strip().split("\n"):
                    # SharedMap prints "Final QAP         : 30792"
                    if "Final QAP" in line:
                        m = re.search(r':\s*(\d+)', line)
                        if m:
                            cli_cost = int(m.group(1))
            else:
                cli_cost = None

            # --- Python ---
            t0 = time.perf_counter()
            r = Decomposition.process_map(
                g, hierarchy=hierarchy, distance=distance,
                mode=config, threads=threads, seed=seed,
                imbalance=0.03,
            )
            py_time = time.perf_counter() - t0
            py_cost = r.comm_cost

            match_str = ""
            if cli_cost is not None:
                if py_cost == cli_cost:
                    match_str = "MATCH"
                else:
                    diff_pct = (py_cost - cli_cost) / max(cli_cost, 1) * 100
                    match_str = f"DIFF cli={cli_cost} ({diff_pct:+.1f}%)"

            print(f"    {config:8s} comm_cost={py_cost:8d}  py={py_time:.3f}s  cli_cost={'N/A' if cli_cost is None else cli_cost}  {match_str}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CHSZLabLib: Python vs CLI Comparison")
    print("=" * 70)
    print(f"Available CLIs:")
    for name, path in [("DynDeltaOrientation", ORIENTATION_CLI),
                        ("DynMatch", DYNMATCH_CLI),
                        ("DynWMIS", DYNWMIS_CLI),
                        ("SharedMap", SHAREDMAP_CLI)]:
        print(f"  {name:25s} {'OK' if os.path.exists(path) else 'MISSING'}")

    benchmark_dyn_orientation()
    benchmark_dyn_matching()
    benchmark_dyn_wmis()
    benchmark_sharedmap()

    print("\n" + "=" * 70)
    print("Done.")
