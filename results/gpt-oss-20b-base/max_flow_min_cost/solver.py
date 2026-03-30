# solver.py
from typing import Any, List
import networkx as nx
import numpy as np


def dict_to_graph(data: dict) -> tuple[nx.DiGraph, int, int]:
    """
    Construct a directed graph suitable for networkx's min‑cost flow routines.
    Nodes are numbered 0 … n-1.
    """
    n = len(data["capacity"])
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges = [
        (i, j, {"capacity": c, "weight": w})
        for i in range(n)
        for j, c in enumerate(data["capacity"][i])
        if c
        for w in [data["cost"][i][j]]
    ]
    G.add_edges_from(edges)
    return G, data["s"], data["t"]


class Solver:
    """
    Solver for the minimum‑weight assignment problem (minimum cost flow with
    integral capacities).  The solution is returned as a dense adjacency matrix
    of flows, compatible with the input format.
    """

    def solve(self, problem: dict[str, Any]) -> List[List[Any]]:
        n = len(problem["capacity"])

        try:
            G, source, sink = dict_to_graph(problem)
            flow_dict = nx.maximum_flow_min_cost(G, source, sink)
        except Exception:
            # In case of any failure, return a zero matrix of the appropriate size.
            return [[0] * n for _ in range(n)]

        # Build dense result from the sparse flow dictionary
        result = np.zeros((n, n), dtype=int)
        for u, out in flow_dict.items():
            for v, f in out.items():
                result[u][v] = f
        return result.tolist()