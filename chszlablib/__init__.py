from chszlablib.graph import Graph
from chszlablib.io import read_metis, write_metis
from chszlablib.decomposition import (
    Decomposition,
    PartitionResult,
    SeparatorResult,
    OrderingResult,
    MincutResult,
    ClusterResult,
    CorrelationClusteringResult,
    MaxCutResult,
    MotifClusterResult,
)
from chszlablib.heistream import HeiStreamPartitioner, StreamPartitionResult
from chszlablib.independence import IndependenceProblems, MISResult, MWISResult
from chszlablib.orientation import Orientation, EdgeOrientationResult
from chszlablib.paths import PathProblems, LongestPathResult

__all__ = [
    # Core
    "Graph",
    "read_metis",
    "write_metis",
    # Namespace classes
    "Decomposition",
    "IndependenceProblems",
    "Orientation",
    "PathProblems",
    # Streaming (stateful class)
    "HeiStreamPartitioner",
    # Result dataclasses
    "PartitionResult",
    "SeparatorResult",
    "OrderingResult",
    "MincutResult",
    "ClusterResult",
    "CorrelationClusteringResult",
    "MaxCutResult",
    "MotifClusterResult",
    "StreamPartitionResult",
    "MISResult",
    "MWISResult",
    "EdgeOrientationResult",
    "LongestPathResult",
]
