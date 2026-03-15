import pytest
from chszlablib import Graph

@pytest.fixture
def simple_path_graph():
    """Path graph: 0--1--2--3 (all weights 1)."""
    g = Graph(num_nodes=4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    return g

@pytest.fixture
def weighted_graph():
    """Small weighted graph for testing."""
    g = Graph(num_nodes=5)
    g.add_edge(0, 1, weight=2)
    g.add_edge(0, 2, weight=3)
    g.add_edge(1, 2, weight=1)
    g.add_edge(1, 3, weight=4)
    g.add_edge(2, 4, weight=5)
    g.add_edge(3, 4, weight=2)
    return g
