from typing import Any, List

import networkx as nx
import numpy as np


def dict_to_graph(data):
    """Build a directed graph from the problem dictionary."""
    n = len(data["capacity"])
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    capacity = data["capacity"]
    cost = data["cost"]
    for i in range(n):
        for j in range(n):
            cap = capacity[i][j]
            if cap:
                G.add_edge(i, j, capacity=cap, weight=cost[i][j])
    return G, data["s"], data["t"]


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[Any]]:
        """
        Minimum cost flow using NetworkX's efficient implementation.
        Returns the flow matrix with no extraneous overhead.
        """
        n = len(problem["capacity"])
        G, s, t = dict_to_graph(problem)

        # Compute min‑cost flow once
        flow_dict = nx.max_flow_min_cost(G, s, t)

        # Pre‑allocate an integer matrix
        result = np.zeros((n, n), dtype=int)

        # Fill only non‑zero entries
        for u, d in flow_dict.items():
            result[u, list(d.keys())] = list(d.values())

        return result.tolist()