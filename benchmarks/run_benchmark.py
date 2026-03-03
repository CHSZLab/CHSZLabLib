#!/usr/bin/env python3
"""Benchmark: CHSZLabLib Python API vs original C++ CLI tools.

Compares solution quality and running time for:
  - CluStRE   (streaming graph clustering)
  - HeiCut    (exact hypergraph minimum cut)
  - HeiHGM    (static b-matching)
  - HeiHGM    (streaming b-matching)
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_repositories"

# CLI binaries
CLUSTRE_BIN = EXT / "CluStRE" / "deploy" / "clustre"
HEICUT_DIR = EXT / "HeiCut"
HEICUT_KERNELIZER = HEICUT_DIR / "build_bench" / "kernelizer"
HEICUT_SUBMODULAR = HEICUT_DIR / "build_bench" / "submodular"
BMATCHING_BIN = EXT / "HeiHGM_Bmatching" / "bazel-bin" / "app" / "app"
STREAMING_BIN = EXT / "HeiHGM_Streaming" / "bazel-bin" / "app" / "app"

# TBB library for HeiCut
HEICUT_TBB = str(HEICUT_DIR / "extern" / "mt-kahypar-library" / "tbb_lib" / "intel64" / "gcc4.8")
HEICUT_MTKAHYPAR = str(HEICUT_DIR / "extern" / "mt-kahypar-library")

# ── Instances ────────────────────────────────────────────────────────────────
GRAPH_INSTANCES = {
    "delaunay_n15": EXT / "CluStRE" / "examples" / "delaunay_n15.graph",
    "astro-ph": EXT / "CluStRE" / "extern" / "VieClus" / "examples" / "astro-ph.graph",
    "as-22july06": EXT / "CluStRE" / "extern" / "VieClus" / "examples" / "as-22july06.graph",
}

HGR_BASE = EXT / "HeidelbergMotifClustering" / "experimental_data" / "Local Motif Clustering via (Hyper)Graph Partitioning" / "full_algorithms" / "extern" / "mt-kahypar" / "tests" / "instances"
HYPERGRAPH_INSTANCES = {
    "ibm01": HGR_BASE / "ibm01.hgr",
    "powersim": HGR_BASE / "powersim.mtx.hgr",
    "sat14_atco": HGR_BASE / "sat14_atco_enc1_opt2_10_16.cnf.primal.hgr",
    "G67": EXT / "HeiCut" / "examples" / "G67.mtx.hgr.random100",
    "cryg10000": EXT / "HeiCut" / "examples" / "cryg10000.mtx.hgr.random100",
}

NRUNS = 3  # number of runs for timing


# ── Helpers ──────────────────────────────────────────────────────────────────
def median_time(times):
    return median(times) if times else float("nan")


def run_cli(cmd, env=None, timeout=600):
    """Run CLI command, return (stdout, wall_clock_seconds)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=full_env, timeout=timeout
    )
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  CLI ERROR (rc={result.returncode}): {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
    return result.stdout + result.stderr, wall


def parse_kv(text, key):
    """Parse 'key<whitespace>value' or 'key: value' from CLI output."""
    for line in text.splitlines():
        stripped = line.strip()
        # Handle "Key: Value" format (CluStRE)
        if stripped.startswith(key + ":"):
            val = stripped[len(key) + 1:].strip()
            if val:
                return val
        # Handle "key\t\t\tvalue" format (HeiCut)
        parts = stripped.split()
        if len(parts) >= 2 and parts[0] == key:
            return parts[-1]
    return None


# ── CluStRE Benchmark ───────────────────────────────────────────────────────
def bench_clustre():
    print("\n=== CluStRE: Streaming Graph Clustering ===")
    from chszlablib import read_metis, Decomposition

    results = []
    for name, path in GRAPH_INSTANCES.items():
        if not path.exists():
            print(f"  SKIP {name}: file not found")
            continue
        for mode in ("light", "strong"):
            print(f"  {name} / {mode} ...", end=" ", flush=True)

            # ── Python API ──
            g = read_metis(path)
            py_times = []
            py_mod = None
            py_ncl = None
            for _ in range(NRUNS):
                t0 = time.perf_counter()
                r = Decomposition.stream_cluster(g, mode=mode, seed=0)
                py_times.append(time.perf_counter() - t0)
                py_mod = r.modularity
                py_ncl = r.num_clusters

            # ── CLI ──
            cli_algo_times = []
            cli_wall_times = []
            cli_mod = None
            cli_ncl = None
            if CLUSTRE_BIN.exists():
                for _ in range(NRUNS):
                    out, wall = run_cli([
                        str(CLUSTRE_BIN), str(path),
                        f"--mode={mode}", "--seed=0",
                        "--suppress_file_output",
                        "--one_pass_algorithm=modularity",
                        "--evaluate",
                    ])
                    cli_wall_times.append(wall)
                    t = parse_kv(out, "Total Time")
                    if t:
                        try:
                            cli_algo_times.append(float(t))
                        except ValueError:
                            pass
                    s = parse_kv(out, "Score")
                    if s:
                        try:
                            cli_mod = float(s)
                        except ValueError:
                            pass
                    c = parse_kv(out, "Clusters Amount")
                    if c:
                        try:
                            cli_ncl = int(c)
                        except ValueError:
                            pass
            else:
                print("(CLI not found)", end=" ")

            row = {
                "instance": name,
                "mode": mode,
                "py_modularity": py_mod,
                "cli_modularity": cli_mod,
                "py_clusters": py_ncl,
                "cli_clusters": cli_ncl,
                "py_time_s": median_time(py_times),
                "cli_algo_time_s": median_time(cli_algo_times) if cli_algo_times else None,
                "cli_wall_time_s": median_time(cli_wall_times) if cli_wall_times else None,
            }
            results.append(row)
            print(f"py_mod={py_mod:.4f}  cli_mod={cli_mod}  py_t={median_time(py_times):.3f}s")

    return results


# ── HeiCut Benchmark ─────────────────────────────────────────────────────────
def bench_heicut():
    print("\n=== HeiCut: Exact Hypergraph Minimum Cut ===")
    from chszlablib import read_hmetis, Decomposition

    heicut_env = {
        "LD_LIBRARY_PATH": f"{HEICUT_MTKAHYPAR}:{HEICUT_TBB}",
    }

    results = []
    for name, path in HYPERGRAPH_INSTANCES.items():
        if not path.exists():
            print(f"  SKIP {name}: file not found")
            continue
        for algo in ("submodular", "kernelizer"):
            print(f"  {name} / {algo} ...", end=" ", flush=True)

            # ── Python API ──
            hg = read_hmetis(path)
            py_times = []
            py_cut = None
            py_algo_time = None
            for _ in range(NRUNS):
                t0 = time.perf_counter()
                r = Decomposition.hypergraph_mincut(hg, algorithm=algo, seed=0)
                py_times.append(time.perf_counter() - t0)
                py_cut = r.cut_value
                py_algo_time = r.time

            # ── CLI ──
            cli_algo_times = []
            cli_wall_times = []
            cli_cut = None
            cli_bin = HEICUT_KERNELIZER if algo == "kernelizer" else HEICUT_SUBMODULAR
            if cli_bin.exists():
                for _ in range(NRUNS):
                    out, wall = run_cli(
                        [str(cli_bin), str(path), "--seed", "0"],
                        env=heicut_env,
                    )
                    cli_wall_times.append(wall)
                    ct = parse_kv(out, "total_computing_time")
                    if ct:
                        try:
                            cli_algo_times.append(float(ct))
                        except ValueError:
                            pass
                    cv = parse_kv(out, "final_mincut_value")
                    if cv:
                        try:
                            cli_cut = int(cv)
                        except ValueError:
                            pass
            else:
                print("(CLI not found)", end=" ")

            row = {
                "instance": name,
                "algorithm": algo,
                "py_cut_value": py_cut,
                "cli_cut_value": cli_cut,
                "quality_match": py_cut == cli_cut if (py_cut is not None and cli_cut is not None) else None,
                "py_time_s": median_time(py_times),
                "py_algo_time_s": py_algo_time,
                "cli_algo_time_s": median_time(cli_algo_times) if cli_algo_times else None,
                "cli_wall_time_s": median_time(cli_wall_times) if cli_wall_times else None,
            }
            results.append(row)
            match_str = "MATCH" if row["quality_match"] else ("MISMATCH" if row["quality_match"] is False else "N/A")
            print(f"py_cut={py_cut}  cli_cut={cli_cut}  [{match_str}]  py_t={median_time(py_times):.3f}s")

    return results


# ── HeiHGM B-Matching Benchmark ─────────────────────────────────────────────
def bench_bmatching():
    print("\n=== HeiHGM: Static B-Matching ===")
    from chszlablib import read_hmetis, IndependenceProblems

    # Map Python algorithm names to CLI protobuf config.
    # Values are either a single (algo, params) tuple or a list of them for chained configs.
    # CLI uses capacity: -1 (node weights as capacities)
    algo_map = {
        "greedy_weight_desc": ('greedy', {'string_params': {'ordering_method': 'bweight'}}),
        "reductions": ('reductions', {'string_params': {'assume_sorted': 'true'}}),
        "ils": [
            ('greedy', {'string_params': {'ordering_method': 'bweight'}}),
            ('ils', {'int64_params': {'max_tries': 15}}),
        ],
    }

    results = []
    for name, path in HYPERGRAPH_INSTANCES.items():
        if not path.exists():
            print(f"  SKIP {name}: file not found")
            continue

        # Read header to get counts for protobuf
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('%') and not line.startswith('c'):
                    parts = line.split()
                    num_edges = int(parts[0])
                    num_nodes = int(parts[1])
                    break

        for py_algo, cli_cfg in algo_map.items():
            print(f"  {name} / {py_algo} ...", end=" ", flush=True)

            # ── Python API ──
            # Use node_weights as capacities to match CLI capacity: -1
            hg = read_hmetis(path)
            if hg.node_weights is not None:
                hg._capacities = hg.node_weights.astype(np.int32)
            py_times = []
            py_weight = None
            py_matched = None
            for _ in range(NRUNS):
                t0 = time.perf_counter()
                # CLI app never sets seed → mt19937 default. Use seed=-1 to skip setSeed.
                bm_seed = -1 if py_algo == "ils" else 0
                r = IndependenceProblems.bmatching(hg, algorithm=py_algo, seed=bm_seed)
                py_times.append(time.perf_counter() - t0)
                py_weight = r.total_weight
                py_matched = r.num_matched

            # ── CLI ──
            cli_wall_times = []
            cli_weight = None
            cli_matched = None
            if BMATCHING_BIN.exists():
                # Build algorithm_configs block(s) — supports single or chained configs
                cfg_steps = cli_cfg if isinstance(cli_cfg, list) else [cli_cfg]
                algo_cfgs = ""
                for cli_algo, cli_params in cfg_steps:
                    algo_cfg = f'algorithm_name: "{cli_algo}"'
                    for ptype, params in cli_params.items():
                        for k, v in params.items():
                            if ptype == "string_params":
                                algo_cfg += f' string_params {{ key: "{k}" value: "{v}" }}'
                            elif ptype == "int64_params":
                                algo_cfg += f' int64_params {{ key: "{k}" value: {v} }}'
                    algo_cfgs += f' algorithm_configs {{ {algo_cfg} }}'

                textproto = (
                    f'command: "run" '
                    f'hypergraph {{ name: "{name}" file_path: "{path}" format: "hgr" '
                    f'node_count: {num_nodes} edge_count: {num_edges} }} '
                    f'config {{ capacity: -1{algo_cfgs} }}'
                )
                for _ in range(NRUNS):
                    out, wall = run_cli(
                        [str(BMATCHING_BIN), f"--command_textproto={textproto}"],
                        timeout=300,
                    )
                    cli_wall_times.append(wall)
                    # Parse protobuf text output
                    w_match = re.search(r'weight:\s*([\d.]+)', out)
                    s_match = re.search(r'size:\s*(\d+)', out)
                    if w_match:
                        cli_weight = float(w_match.group(1))
                    if s_match:
                        cli_matched = int(s_match.group(1))
            else:
                print("(CLI not found)", end=" ")

            row = {
                "instance": name,
                "algorithm": py_algo,
                "py_weight": py_weight,
                "cli_weight": cli_weight,
                "py_matched": py_matched,
                "cli_matched": cli_matched,
                "py_time_s": median_time(py_times),
                "cli_wall_time_s": median_time(cli_wall_times) if cli_wall_times else None,
            }
            results.append(row)
            print(f"py_w={py_weight}  cli_w={cli_weight}  py_t={median_time(py_times):.3f}s")

    return results


# ── HeiHGM Streaming B-Matching Benchmark ───────────────────────────────────
def bench_streaming():
    print("\n=== HeiHGM: Streaming B-Matching ===")
    from chszlablib import read_hmetis, StreamingBMatcher

    algo_map = {
        "naive": "naive",
        "greedy": "greedy",
        "greedy_set": "greedy_set",
    }

    results = []
    for name, path in HYPERGRAPH_INSTANCES.items():
        if not path.exists():
            print(f"  SKIP {name}: file not found")
            continue

        # Read header for node/edge counts
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('%') and not line.startswith('c'):
                    parts = line.split()
                    num_edges = int(parts[0])
                    num_nodes = int(parts[1])
                    break

        # Read edges from hgr file for Python streaming API
        hg = read_hmetis(path)
        edges = []
        for i in range(hg.num_edges):
            start = hg.eptr[i]
            end = hg.eptr[i + 1]
            nodes = hg.everts[start:end].tolist()
            weight = float(hg.edge_weights[i]) if hg.edge_weights is not None else 1.0
            edges.append((nodes, weight))

        for py_algo, cli_algo in algo_map.items():
            print(f"  {name} / {py_algo} ...", end=" ", flush=True)

            # ── Python API ──
            py_times = []
            py_weight = None
            py_matched = None
            for _ in range(NRUNS):
                sm = StreamingBMatcher(num_nodes, algorithm=py_algo, seed=0)
                t0 = time.perf_counter()
                for nodes, weight in edges:
                    sm.add_edge(nodes, weight)
                r = sm.finish()
                py_times.append(time.perf_counter() - t0)
                py_weight = r.total_weight
                py_matched = r.num_matched

            # ── CLI ──
            cli_wall_times = []
            cli_weight = None
            cli_matched = None
            if STREAMING_BIN.exists():
                textproto = (
                    f'command: "run" '
                    f'hypergraph {{ name: "{name}" file_path: "{path}" format: "hgr" '
                    f'node_count: {num_nodes} edge_count: {num_edges} }} '
                    f'config {{ capacity: 1 seeds: 0 '
                    f'algorithm_configs {{ data_structure: "from_disk_stream_hypergraph" '
                    f'algorithm_name: "{cli_algo}" }} }}'
                )
                for _ in range(NRUNS):
                    out, wall = run_cli(
                        [str(STREAMING_BIN), f"--command_textproto={textproto}", "--seed=0"],
                        timeout=300,
                    )
                    cli_wall_times.append(wall)
                    w_match = re.search(r'weight:\s*([\d.]+)', out)
                    s_match = re.search(r'size:\s*(\d+)', out)
                    if w_match:
                        cli_weight = float(w_match.group(1))
                    if s_match:
                        cli_matched = int(s_match.group(1))
            else:
                print("(CLI not found)", end=" ")

            row = {
                "instance": name,
                "algorithm": py_algo,
                "py_weight": py_weight,
                "cli_weight": cli_weight,
                "py_matched": py_matched,
                "cli_matched": cli_matched,
                "py_time_s": median_time(py_times),
                "cli_wall_time_s": median_time(cli_wall_times) if cli_wall_times else None,
            }
            results.append(row)
            print(f"py_w={py_weight}  cli_w={cli_weight}  py_t={median_time(py_times):.3f}s")

    return results


# ── Report Generation ────────────────────────────────────────────────────────
def system_info():
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    try:
        r = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        info["gcc"] = r.stdout.splitlines()[0] if r.returncode == 0 else "N/A"
    except FileNotFoundError:
        info["gcc"] = "N/A"
    try:
        r = subprocess.run(["nproc"], capture_output=True, text=True)
        info["cores"] = r.stdout.strip()
    except FileNotFoundError:
        info["cores"] = "N/A"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = f"{kb / 1024 / 1024:.1f}"
                    break
    except Exception:
        info["ram_gb"] = "N/A"
    return info


def fmt_time(t):
    if t is None:
        return "N/A"
    if t < 0.001:
        return f"{t*1e6:.0f} us"
    if t < 1:
        return f"{t*1000:.1f} ms"
    return f"{t:.3f} s"


def fmt_val(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def generate_report(clustre_res, heicut_res, bmatching_res, streaming_res):
    info = system_info()
    lines = []
    lines.append("# Benchmark Report: Python API vs C++ CLI\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## System Information\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Platform | {info['platform']} |")
    lines.append(f"| CPU | {info['cpu']} |")
    lines.append(f"| Cores | {info['cores']} |")
    lines.append(f"| RAM | {info['ram_gb']} GB |")
    lines.append(f"| Python | {info['python']} |")
    lines.append(f"| Compiler | {info['gcc']} |")
    lines.append(f"| Runs per config | {NRUNS} (median reported) |")
    lines.append("")

    # CluStRE
    lines.append("## CluStRE: Streaming Graph Clustering\n")
    lines.append("| Instance | Mode | Py Modularity | CLI Modularity | Py Clusters | CLI Clusters | Py Time | CLI Algo Time | CLI Wall Time |")
    lines.append("|----------|------|---------------|----------------|-------------|--------------|---------|---------------|---------------|")
    for r in clustre_res:
        lines.append(
            f"| {r['instance']} | {r['mode']} "
            f"| {fmt_val(r['py_modularity'])} | {fmt_val(r['cli_modularity'])} "
            f"| {r.get('py_clusters', 'N/A')} | {r.get('cli_clusters', 'N/A')} "
            f"| {fmt_time(r['py_time_s'])} | {fmt_time(r.get('cli_algo_time_s'))} "
            f"| {fmt_time(r.get('cli_wall_time_s'))} |"
        )
    lines.append("")

    # HeiCut
    lines.append("## HeiCut: Exact Hypergraph Minimum Cut\n")
    lines.append("| Instance | Algorithm | Py Cut | CLI Cut | Match | Py Time | Py Algo Time | CLI Algo Time | CLI Wall Time |")
    lines.append("|----------|-----------|--------|---------|-------|---------|--------------|---------------|---------------|")
    for r in heicut_res:
        match_str = "yes" if r.get("quality_match") else ("NO" if r.get("quality_match") is False else "N/A")
        lines.append(
            f"| {r['instance']} | {r['algorithm']} "
            f"| {r['py_cut_value']} | {r.get('cli_cut_value', 'N/A')} "
            f"| {match_str} "
            f"| {fmt_time(r['py_time_s'])} | {fmt_time(r.get('py_algo_time_s'))} "
            f"| {fmt_time(r.get('cli_algo_time_s'))} "
            f"| {fmt_time(r.get('cli_wall_time_s'))} |"
        )
    lines.append("")

    # B-Matching
    lines.append("## HeiHGM: Static B-Matching\n")
    lines.append("| Instance | Algorithm | Py Weight | CLI Weight | Py Matched | CLI Matched | Py Time | CLI Wall Time |")
    lines.append("|----------|-----------|-----------|------------|------------|-------------|---------|---------------|")
    for r in bmatching_res:
        lines.append(
            f"| {r['instance']} | {r['algorithm']} "
            f"| {fmt_val(r.get('py_weight'), '.1f')} | {fmt_val(r.get('cli_weight'), '.1f')} "
            f"| {r.get('py_matched', 'N/A')} | {r.get('cli_matched', 'N/A')} "
            f"| {fmt_time(r['py_time_s'])} "
            f"| {fmt_time(r.get('cli_wall_time_s'))} |"
        )
    lines.append("")

    # Streaming
    lines.append("## HeiHGM: Streaming B-Matching\n")
    lines.append("| Instance | Algorithm | Py Weight | CLI Weight | Py Matched | CLI Matched | Py Time | CLI Wall Time |")
    lines.append("|----------|-----------|-----------|------------|------------|-------------|---------|---------------|")
    for r in streaming_res:
        lines.append(
            f"| {r['instance']} | {r['algorithm']} "
            f"| {fmt_val(r.get('py_weight'), '.1f')} | {fmt_val(r.get('cli_weight'), '.1f')} "
            f"| {r.get('py_matched', 'N/A')} | {r.get('cli_matched', 'N/A')} "
            f"| {fmt_time(r['py_time_s'])} "
            f"| {fmt_time(r.get('cli_wall_time_s'))} |"
        )
    lines.append("")

    # Summary
    lines.append("## Summary\n")

    # Quality analysis
    heicut_matches = sum(1 for r in heicut_res if r.get("quality_match"))
    heicut_total = sum(1 for r in heicut_res if r.get("quality_match") is not None)
    lines.append(f"- **HeiCut quality parity:** {heicut_matches}/{heicut_total} configurations produce identical cut values")

    # Timing analysis
    lines.append("- **Timing overhead:** Python API includes:")
    lines.append("  - Graph/hypergraph I/O (reading + building internal data structures)")
    lines.append("  - Python interpreter overhead")
    lines.append("  - pybind11 marshalling (CSR arrays to/from C++)")
    lines.append("- **CLI wall time** includes process startup, file I/O, and argument parsing")
    lines.append("- **CLI algo time** is the algorithm's own reported execution time (excludes I/O)")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Ensure we can import chszlablib
    sys.path.insert(0, str(ROOT))

    print("=" * 60)
    print("  CHSZLabLib Benchmark: Python API vs C++ CLI")
    print("=" * 60)

    # Check which CLI tools are available
    for name, path in [
        ("CluStRE", CLUSTRE_BIN),
        ("HeiCut/kernelizer", HEICUT_KERNELIZER),
        ("HeiCut/submodular", HEICUT_SUBMODULAR),
        ("HeiHGM/Bmatching", BMATCHING_BIN),
        ("HeiHGM/Streaming", STREAMING_BIN),
    ]:
        status = "OK" if path.exists() else "NOT FOUND"
        print(f"  {name}: {status}")

    clustre_res = bench_clustre()
    heicut_res = bench_heicut()
    bmatching_res = bench_bmatching()
    streaming_res = bench_streaming()

    # Generate and save report
    report = generate_report(clustre_res, heicut_res, bmatching_res, streaming_res)
    report_path = ROOT / "report.md"
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")

    # Also save raw JSON
    raw = {
        "clustre": clustre_res,
        "heicut": heicut_res,
        "bmatching": bmatching_res,
        "streaming": streaming_res,
    }
    json_path = ROOT / "benchmarks" / "results.json"
    json_path.write_text(json.dumps(raw, indent=2, default=str))
    print(f"Raw results written to {json_path}")


if __name__ == "__main__":
    main()
