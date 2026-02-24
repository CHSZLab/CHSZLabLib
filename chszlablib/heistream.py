"""Streaming graph partitioning via HeiStream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class StreamPartitionResult:
    """Result of a streaming graph partitioning call."""

    assignment: np.ndarray
    """Partition assignment for each node (0-indexed)."""


class HeiStreamPartitioner:
    """Streaming graph partitioner using HeiStream.

    Supports both the full BuffCut algorithm (with configurable buffer and
    window/batch sizes) and the simpler direct Fennel one-pass mode.

    Usage::

        hs = HeiStreamPartitioner(k=4, imbalance=3.0, max_buffer_size=1000)
        hs.new_node(0, [1, 2])
        hs.new_node(1, [0, 3])
        hs.new_node(2, [0])
        hs.new_node(3, [1])
        result = hs.partition()
        print(result.assignment)   # array of partition IDs

    Parameters
    ----------
    k : int
        Number of partitions.
    imbalance : float
        Allowed imbalance in percent (e.g. 3.0 means 3%).
    seed : int
        Random seed.
    max_buffer_size : int
        Buffer size for BuffCut. Set to 0 or 1 for direct Fennel (no buffer).
        Larger values enable the priority-buffer mode.
    batch_size : int
        MLP batch size for model-based partitioning within the buffer.
        Set to 0 for HeiStream's default.
    num_streams_passes : int
        Number of streaming passes (restreaming).
    run_parallel : bool
        Use the parallel 3-thread pipeline (I/O, PQ, partition).
    suppress_output : bool
        Suppress stdout/stderr from the C++ algorithm.
    """

    def __init__(
        self,
        k: int = 2,
        imbalance: float = 3.0,
        seed: int = 0,
        max_buffer_size: int = 0,
        batch_size: int = 0,
        num_streams_passes: int = 1,
        run_parallel: bool = False,
        suppress_output: bool = True,
    ):
        self._k = k
        self._imbalance = imbalance
        self._seed = seed
        self._max_buffer_size = max_buffer_size
        self._batch_size = batch_size
        self._num_streams_passes = num_streams_passes
        self._run_parallel = run_parallel
        self._suppress_output = suppress_output

        self._nodes: list[list[int]] = []
        self._node_map: dict[int, int] = {}

    def new_node(self, node: int, neighbors: Sequence[int]) -> None:
        """Add a node with its neighborhood to the stream.

        Parameters
        ----------
        node : int
            The node ID (0-indexed).
        neighbors : sequence of int
            Neighbor node IDs (0-indexed).
        """
        if node in self._node_map:
            raise ValueError(f"Node {node} has already been added")
        self._node_map[node] = len(self._nodes)
        self._nodes.append(list(neighbors))

    def partition(self) -> StreamPartitionResult:
        """Run the HeiStream algorithm on all added nodes.

        Returns
        -------
        StreamPartitionResult
            The partition assignment for each node.
        """
        from chszlablib._heistream import heistream_partition

        n = len(self._nodes)
        if n == 0:
            return StreamPartitionResult(assignment=np.array([], dtype=np.int32))

        # Build CSR representation
        # First remap node IDs to contiguous 0..n-1 if needed
        original_ids = sorted(self._node_map.keys())
        id_to_contiguous = {orig: i for i, orig in enumerate(original_ids)}

        xadj = [0]
        adjncy = []
        for orig_id in original_ids:
            idx = self._node_map[orig_id]
            neighbors = self._nodes[idx]
            mapped = []
            for nb in neighbors:
                if nb in id_to_contiguous:
                    mapped.append(id_to_contiguous[nb])
            adjncy.extend(mapped)
            xadj.append(len(adjncy))

        xadj_arr = np.array(xadj, dtype=np.int64)
        adjncy_arr = np.array(adjncy, dtype=np.int64)

        raw_assignment = heistream_partition(
            xadj_arr,
            adjncy_arr,
            k=self._k,
            imbalance=self._imbalance,
            seed=self._seed,
            max_buffer_size=self._max_buffer_size,
            batch_size=self._batch_size,
            num_streams_passes=self._num_streams_passes,
            run_parallel=self._run_parallel,
            suppress_output=self._suppress_output,
        )

        # Map back to original node IDs if non-contiguous
        if original_ids == list(range(n)):
            assignment = raw_assignment
        else:
            assignment = np.full(max(original_ids) + 1, -1, dtype=np.int32)
            for i, orig_id in enumerate(original_ids):
                assignment[orig_id] = raw_assignment[i]

        return StreamPartitionResult(assignment=assignment)

    def reset(self) -> None:
        """Clear all nodes to reuse this partitioner."""
        self._nodes.clear()
        self._node_map.clear()


