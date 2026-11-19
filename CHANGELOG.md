# Changelog

All notable changes to CHSZLabLib are documented in this file. Versions before 0.6.1 are listed on the [GitHub tags page](https://github.com/CHSZLab/CHSZLabLib/tags).

## [0.6.1] (unreleased)

### Added

- **Connected blocks in graph partitioning:** `Decomposition.partition(..., connected_blocks=True)` and `Decomposition.evolutionary_partition(..., connected_blocks=True)` make every block a connected subgraph (if the input graph is connected). Requires mode `"strong"`; other modes raise `ValueError`. Uses `kaffpa_connected_blocks`, new in KaHIP v3.29.
- **KaHIP `ultra` mode:** `mode="ultra"` for `Decomposition.partition` and `Decomposition.evolutionary_partition`. It is a higher-quality preconfiguration that starts from `strong` and adds Jet refinement, the unlocked flow model and more global work (KaHIP v3.28).
- **Stoer-Wagner minimum cut:** `Decomposition.mincut(g, algorithm="sw")` (VieCut v2.0, where Stoer-Wagner was fixed: it previously returned wrong cuts or crashed).
- **Leiden clustering:** `Decomposition.cluster(g, leiden=True, leiden_theta=0.01)` uses the Leiden algorithm instead of Louvain inside VieClus, so every cluster is a connected subgraph (VieClus v1.3.1).

### Performance

The KaHIP and VieCut speedups below were engineered with [Agentic Algorithm Engineering](https://github.com/CHSZLab/AgenticAlgorithmEngineering) (AAE, [paper](https://arxiv.org/abs/2609.07204)): a loop of profiling, hypothesis, implementation and measurement on a fixed benchmark.

- **Graph partitioning (KaHIP v3.26):** `kaffpa` is 3.9x faster in total time and 2.8x faster in geometric mean over a benchmark of 537 runs (20 mesh, road and social/web graphs, all six preconfigurations, k = 2 to 256), with partition quality unchanged. This speeds up `Decomposition.partition` and the partitioning inside the other KaHIP-based functions.
- **`ultra` mode (KaHIP v3.28):** also developed with AAE, with a focus on solution quality: 2.2% smaller cuts than v3.26 `strong` in geometric mean (about 7% on social graphs) at 2.0x its time.
- **Exact minimum cut (VieCut v2.0):** 8.9x faster on one thread and 8.8x faster on 32 threads (geometric mean over a 53-instance benchmark), with every computed cut unchanged. Large-cut instances no longer run out of memory. This speeds up `Decomposition.mincut` with `"exact"` and `"noi"`.
- **Streaming hypergraph partitioning (FREIGHT v1.0.3):** multi-pass streaming about 4.3x faster.

### Changed

- **`ULTRAFASTSOCIAL` renumbered from 6 to 7** in the `_kahipe` module, because KaHIP v3.28 uses 6 for `ULTRA`. Mode strings (`"ultrafastsocial"`) are unaffected; only code that passes the raw integer must be updated.
- **HyperMIS:** unconfined vertex reductions are always active and the strong reduction settings are the upstream defaults (HyperMIS v1.1). The public `IndependenceProblems.hypermis()` API is unchanged.

### Fixed

- **README and `docs/hypergraph-independent-set.md`:** the `hypermis` documentation now matches the actual API, `hypermis(hg, time_limit=60.0, seed=0)` (reductions, then an exact ILP on the kernel via `gurobipy`). The `method` and `strong_reductions` parameters it described do not exist.

### Updated libraries

| Library | Previous | Now | Highlights |
|:--------|:---------|:----|:-----------|
| KaHIP | v3.22 (+2 commits) | v3.29 | All changes of KaHIP v3.23 to v3.29, including `--connected_blocks` (v3.25), 3.9x faster KaFFPa via AAE (v3.26), the `ultra` preconfiguration via AAE and a flow refinement bug fix (v3.28), and connected blocks in the library (v3.29); see the [KaHIP releases](https://github.com/KaHIP/KaHIP/releases) |
| VieCut | earlier commit | v2.0 | Exact minimum cut 8.9x faster via AAE, no out-of-memory on large cuts, Stoer-Wagner fixed; see the [VieCut v2.0 release](https://github.com/KaHIP/VieCut/releases/tag/v2.0) |
| VieClus | v1.2.0 | v1.3.1 | Leiden mode |
| SCC | earlier commit | v1.3 | Performance optimizations, double edge weights as a compile-time option |
| FREIGHT | v1.0.1 | v1.0.3 | Multi-pass performance optimizations (about 4.3x faster) |
| HeiCut | v1.0.5 | v1.0.7 | Kernelizer and solver updates, minor fixes |
| HyperMIS | Arxiv2026 | v1.1 | Edge domination and unconfined reduction bug fixes |
| CHILS | SEA2025 (+91 commits) | v1.2 | Initial solution bug fix |
