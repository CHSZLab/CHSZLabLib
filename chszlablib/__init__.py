from chszlablib.graph import Graph
from chszlablib.io import read_metis, write_metis
from chszlablib.decomposition import (
    Decomposition,
    HeiStreamPartitioner,
    PartitionResult,
    SeparatorResult,
    OrderingResult,
    MincutResult,
    ClusterResult,
    CorrelationClusteringResult,
    MaxCutResult,
    MotifClusterResult,
    StreamPartitionResult,
)
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
    "HeiStreamPartitioner",
    "IndependenceProblems",
    "Orientation",
    "PathProblems",
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
