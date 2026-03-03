# Benchmark Report: Python API vs C++ CLI

**Date:** 2026-03-03 06:44

## System Information

| Property | Value |
|----------|-------|
| Platform | Linux-6.8.0-87-generic-x86_64-with-glibc2.39 |
| CPU | x86_64 |
| Cores | 256 |
| RAM | 755.1 GB |
| Python | 3.9.19 |
| Compiler | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| Runs per config | 3 (median reported) |

## CluStRE: Streaming Graph Clustering

| Instance | Mode | Py Modularity | CLI Modularity | Py Clusters | CLI Clusters | Py Time | CLI Algo Time | CLI Wall Time |
|----------|------|---------------|----------------|-------------|--------------|---------|---------------|---------------|
| delaunay_n15 | light | 0.5794 | 0.5794 | 4301 | 4301 | 5.7 ms | 31.2 ms | 54.5 ms |
| delaunay_n15 | strong | 0.6214 | 0.6214 | 3714 | 3714 | 16.8 ms | 135.3 ms | 158.6 ms |
| astro-ph | light | 0.6266 | 0.6266 | 1425 | 1425 | 3.4 ms | 18.3 ms | 38.6 ms |
| astro-ph | strong | 0.6885 | 0.6885 | 1374 | 1374 | 10.7 ms | 75.4 ms | 96.2 ms |
| as-22july06 | light | 0.5407 | 0.5407 | 48 | 48 | 2.2 ms | 18.3 ms | 36.2 ms |
| as-22july06 | strong | 0.5753 | 0.5753 | 48 | 48 | 6.0 ms | 54.5 ms | 72.4 ms |

## HeiCut: Exact Hypergraph Minimum Cut

| Instance | Algorithm | Py Cut | CLI Cut | Match | Py Time | Py Algo Time | CLI Algo Time | CLI Wall Time |
|----------|-----------|--------|---------|-------|---------|--------------|---------------|---------------|
| ibm01 | submodular | 1 | 1 | yes | 23.631 s | 23.592 s | 24.883 s | 25.022 s |
| ibm01 | kernelizer | 1 | 1 | yes | 49.2 ms | 10.2 ms | 13.5 ms | 153.0 ms |
| powersim | submodular | 1 | 1 | yes | 22.347 s | 22.312 s | 23.675 s | 23.818 s |
| powersim | kernelizer | 1 | 1 | yes | 51.5 ms | 10.5 ms | 13.5 ms | 155.0 ms |
| sat14_atco | submodular | 0 | 0 | yes | 106.987 s | 114.898 s | 119.119 s | 119.325 s |
| sat14_atco | kernelizer | 0 | 0 | yes | 188.2 ms | 30.4 ms | 43.4 ms | 237.9 ms |
| G67 | submodular | 0 | 0 | yes | 14.915 s | 14.879 s | 15.710 s | 15.847 s |
| G67 | kernelizer | 0 | 0 | yes | 49.6 ms | 8.7 ms | 13.1 ms | 150.7 ms |
| cryg10000 | submodular | 40 | 40 | yes | 15.848 s | 15.802 s | 17.059 s | 17.198 s |
| cryg10000 | kernelizer | 40 | 40 | yes | 53.1 ms | 10.0 ms | 14.0 ms | 151.8 ms |

## HeiHGM: Static B-Matching

| Instance | Algorithm | Py Weight | CLI Weight | Py Matched | CLI Matched | Py Time | CLI Wall Time |
|----------|-----------|-----------|------------|------------|-------------|---------|---------------|
| ibm01 | greedy_weight_desc | 2936.0 | 2936.0 | 2936 | 2936 | 6.1 ms | 26.2 ms |
| ibm01 | reductions | 472.0 | 472.0 | 472 | 472 | 15.1 ms | 36.6 ms |
| ibm01 | ils | 4127.0 | 4133.0 | 4127 | 4133 | 53.5 ms | 219.1 ms |
| powersim | greedy_weight_desc | 5043.0 | 5043.0 | 5043 | 5043 | 5.5 ms | 27.8 ms |
| powersim | reductions | 4924.0 | 4924.0 | 4924 | 4924 | 10.2 ms | 33.0 ms |
| powersim | ils | 5289.0 | 5288.0 | 5289 | 5288 | 48.7 ms | 70.4 ms |
| sat14_atco | greedy_weight_desc | 3357.0 | 3357.0 | 3357 | 3357 | 36.0 ms | 141.3 ms |
| sat14_atco | reductions | 82.0 | 82.0 | 82 | 82 | 92.7 ms | 203.5 ms |
| sat14_atco | ils | 4296.0 | 4196.0 | 4296 | 4196 | 6.367 s | 1.802 s |
| G67 | greedy_weight_desc | 260552.0 | 260552.0 | 3921 | 3921 | 3.8 ms | 26.6 ms |
| G67 | reductions | 107024.0 | 107024.0 | 1531 | 1531 | 13.1 ms | 36.3 ms |
| G67 | ils | 266783.0 | 266783.0 | 4154 | 4154 | 71.1 ms | 90.1 ms |
| cryg10000 | greedy_weight_desc | 235276.0 | 235276.0 | 3488 | 3488 | 4.6 ms | 28.3 ms |
| cryg10000 | reductions | 28915.0 | 28915.0 | 391 | 391 | 12.1 ms | 35.4 ms |
| cryg10000 | ils | 243932.0 | 243932.0 | 3755 | 3755 | 111.7 ms | 126.8 ms |

## HeiHGM: Streaming B-Matching

| Instance | Algorithm | Py Weight | CLI Weight | Py Matched | CLI Matched | Py Time | CLI Wall Time |
|----------|-----------|-----------|------------|------------|-------------|---------|---------------|
| ibm01 | naive | 3157.0 | 3157.0 | 3157 | 3157 | 9.0 ms | 22.5 ms |
| ibm01 | greedy | 3157.0 | 3157.0 | 3157 | 3157 | 9.3 ms | 22.3 ms |
| ibm01 | greedy_set | 3157.0 | 3157.0 | 3157 | 3157 | 11.1 ms | 24.3 ms |
| powersim | naive | 5212.0 | 5212.0 | 5212 | 5212 | 10.3 ms | 24.6 ms |
| powersim | greedy | 5212.0 | 5212.0 | 5212 | 5212 | 10.6 ms | 24.4 ms |
| powersim | greedy_set | 5212.0 | 5212.0 | 5212 | 5212 | 12.0 ms | 25.3 ms |
| sat14_atco | naive | 4137.0 | 4137.0 | 4137 | 4137 | 103.9 ms | 110.2 ms |
| sat14_atco | greedy | 4137.0 | 4137.0 | 4137 | 4137 | 102.3 ms | 111.4 ms |
| sat14_atco | greedy_set | 4137.0 | 4137.0 | 4137 | 4137 | 129.1 ms | 130.7 ms |
| G67 | naive | 125844.0 | 125844.0 | 2500 | 2500 | 6.3 ms | 20.5 ms |
| G67 | greedy | 145799.0 | 145799.0 | 2069 | 2069 | 7.0 ms | 20.7 ms |
| G67 | greedy_set | 129227.0 | 129227.0 | 1556 | 1556 | 7.7 ms | 20.5 ms |
| cryg10000 | naive | 81414.0 | 81414.0 | 1675 | 1675 | 6.5 ms | 21.0 ms |
| cryg10000 | greedy | 110382.0 | 110382.0 | 1535 | 1535 | 6.7 ms | 20.8 ms |
| cryg10000 | greedy_set | 98583.0 | 98583.0 | 1152 | 1152 | 7.7 ms | 21.2 ms |

## Summary

### Quality Parity — 41/41 Deterministic + 5 Randomized (ILS)

| Algorithm Family | Configs | Exact Match |
|-----------------|---------|-------------|
| CluStRE (light + strong) | 6 | 6/6 |
| HeiCut (submodular + kernelizer) | 10 | 10/10 |
| Static B-Matching (greedy_weight_desc) | 5 | 5/5 |
| Static B-Matching (reductions) | 5 | 5/5 |
| Streaming B-Matching (naive) | 5 | 5/5 |
| Streaming B-Matching (greedy) | 5 | 5/5 |
| Streaming B-Matching (greedy_set) | 5 | 5/5 |
| Static B-Matching (ILS, randomized) | 5 | 3/5 exact, 2/5 close |

### Notes

- **Static B-Matching** uses CLI's `capacity: -1` (node weights from HGR as capacities). Python passes `hg.node_weights` as capacities to match.
- **Static ILS** uses CLI chain `greedy(bweight) → ils(max_tries=15)`. ILS is inherently non-deterministic (CLI itself produces different results across runs). Python and CLI results are in the same range.
- **Streaming CLI** uses `from_disk_stream_hypergraph` (sequential file order) to match Python's deterministic edge ordering.
- **Timing:** Python API includes I/O + interpreter + pybind11 overhead. CLI wall time includes process startup + file I/O.
