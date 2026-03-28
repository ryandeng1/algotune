import itertools
import networkx as nx
import numpy as np
from typing import List

def solve(problem: List[List[float]]) -> List[int]:
    """
    Solve the BTSP problem using a fast approximation.
    Returns a tour as a list of city indices starting and ending at city 0.
    """
    n = len(problem)
    if n <= 1:
        return [0, 0]

    # Build graph
    G = nx.Graph()
    for i, j in itertools.combinations(range(n), 2):
        G.add_edge(i, j, weight=problem[i][j])

    # Approximate solution via Christofides
    cycle = nx.approximation.traveling_salesman.christofides(G)
    # Convert cycle (list of nodes) to edges
    cycle = list(cycle)
    edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    # Create tour graph and get DFS order starting at 0
    tour_graph = nx.Graph()
    tour_graph.add_edges_from(edges)
    path = list(nx.dfs_preorder_nodes(tour_graph, source=0))
    path.append(0)
    return path