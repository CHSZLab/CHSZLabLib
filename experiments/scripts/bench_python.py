#!/usr/bin/env python3
"""Benchmark CHSZLabLib Python interface — measures graph I/O + algorithm time."""

import argparse
import json
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from chszlablib import read_metis, Decomposition, IndependenceProblems
from chszlablib.graph import Graph


def bench_partition(g, mode="eco", k=2, seed=0, imbalance=0.03):
    t0 = time.perf_counter()
    r = Decomposition.partition(g, num_parts=k, mode=mode, imbalance=imbalance, seed=seed, suppress_output=True)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "edgecut": int(r.edgecut), "metric_name": "edgecut", "metric_value": int(r.edgecut)}


def bench_mincut(g, algorithm="inexact", seed=0):
    t0 = time.perf_counter()
    r = Decomposition.mincut(g, algorithm=algorithm, seed=seed)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "cut_value": int(r.cut_value), "metric_name": "cut_value", "metric_value": int(r.cut_value)}


def bench_cluster(g, time_limit=1.0, seed=0):
    t0 = time.perf_counter()
    r = Decomposition.cluster(g, time_limit=time_limit, seed=seed, suppress_output=True)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "modularity": float(r.modularity), "num_clusters": int(r.num_clusters),
            "metric_name": "modularity", "metric_value": float(r.modularity)}


def bench_corr_clustering(g, seed=0):
    t0 = time.perf_counter()
    r = Decomposition.correlation_clustering(g, seed=seed)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "edge_cut": int(r.edge_cut), "num_clusters": int(r.num_clusters),
            "metric_name": "edge_cut", "metric_value": int(r.edge_cut)}


def bench_redumis(g, time_limit=10.0, seed=0):
    t0 = time.perf_counter()
    r = IndependenceProblems.redumis(g, time_limit=time_limit, seed=seed)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "size": int(r.size), "weight": int(r.weight),
            "metric_name": "mis_size", "metric_value": int(r.size)}


def bench_online_mis(g, time_limit=10.0, seed=0):
    t0 = time.perf_counter()
    r = IndependenceProblems.online_mis(g, time_limit=time_limit, seed=seed)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "size": int(r.size), "weight": int(r.weight),
            "metric_name": "mis_size", "metric_value": int(r.size)}


def bench_chils(g, time_limit=10.0, seed=0, num_concurrent=4):
    t0 = time.perf_counter()
    r = IndependenceProblems.chils(g, time_limit=time_limit, seed=seed, num_concurrent=num_concurrent)
    dt = time.perf_counter() - t0
    return {"algo_time": dt, "size": int(r.size), "weight": int(r.weight),
            "metric_name": "mwis_weight", "metric_value": int(r.weight)}


def bench_stream_partition(g, k=2, seed=0, imbalance=3.0):
    t0 = time.perf_counter()
    r = Decomposition.stream_partition(g, k=k, seed=seed, imbalance=imbalance, suppress_output=True)
    dt = time.perf_counter() - t0
    # Compute edge cut from assignment (vectorized)
    xadj, adjncy = g._xadj, g._adjncy
    assignment = r.assignment
    src = np.repeat(np.arange(g.num_nodes), np.diff(xadj))
    cut = int(np.sum(assignment[src] != assignment[adjncy])) // 2
    return {"algo_time": dt, "edgecut": cut, "metric_name": "edgecut", "metric_value": cut}


BENCHMARKS = {
    "partition": bench_partition,
    "mincut": bench_mincut,
    "cluster": bench_cluster,
    "corr_clustering": bench_corr_clustering,
    "redumis": bench_redumis,
    "online_mis": bench_online_mis,
    "chils": bench_chils,
    "stream_partition": bench_stream_partition,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--algorithm", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--params", type=str, default="{}", help="JSON dict of algo params")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params = json.loads(args.params)
    params.setdefault("seed", args.seed)

    graph_name = os.path.basename(args.graph).replace(".graph", "")

    # Time graph loading
    t_load_start = time.perf_counter()
    g = read_metis(args.graph)
    t_load = time.perf_counter() - t_load_start

    # Run algorithm
    total_start = time.perf_counter()
    result = BENCHMARKS[args.algorithm](g, **params)
    total_time = time.perf_counter() - total_start

    output = {
        "interface": "python",
        "algorithm": args.algorithm,
        "graph": graph_name,
        "n": g.num_nodes,
        "m": g.num_edges,
        "seed": args.seed,
        "params": params,
        "load_time": t_load,
        "algo_time": result["algo_time"],
        "total_time": total_time + t_load,
        "metric_name": result["metric_name"],
        "metric_value": result["metric_value"],
    }
    # Add extra result fields
    for k, v in result.items():
        if k not in ("algo_time", "metric_name", "metric_value"):
            output[k] = v

    print(json.dumps(output))


if __name__ == "__main__":
    main()
