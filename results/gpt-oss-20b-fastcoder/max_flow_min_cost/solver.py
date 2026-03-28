from typing import Any
import networkx as nx
import numpy as np


def dict_to_graph(data):
    cap = data['capacity']
    cost = data['cost']
    n = len(cap)
    G = nx.DiGraph()
    edges = []
    for i in range(n):
        for j, c in enumerate(cap[i]):
            if c:
                edges.append((i, j, {'capacity': c, 'weight': cost[i][j]}))
    G.add_edges_from(edges)
    return G, data['s'], data['t']


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[Any]]:
        n = len(problem['capacity'])
        G, s, t = dict_to_graph(problem)
        try:
            flow_dict = nx.maximum_flow_min_cost(G, s, t)[1]
        except Exception:
            return [[0] * n for _ in range(n)]

        # Build matrix from flow_dict
        sol = [[0] * n for _ in range(n)]
        for i, out in flow_dict.items():
            for j, f in out.items():
                sol[i][j] = f
        return sol