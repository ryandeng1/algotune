from typing import Any
import networkx as nx


def dict_to_graph(data):
    capacity = data["capacity"]
    cost = data["cost"]
    s_idx = data["s"]
    t_idx = data["t"]
    n = len(capacity)

    G = nx.DiGraph()
    edges = []
    for i in range(n):
        for j in range(n):
            if capacity[i][j] > 0:
                edges.append((i, j, {'capacity': capacity[i][j], 'cost': cost[i][j]}))
    G.add_edges_from(edges)
    return G, s_idx, t_idx


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[Any]]:
        try:
            n = len(problem["capacity"])
            G, s, t = dict_to_graph(problem)
            mincostFlow = nx.max_flow_min_cost(G, s, t)
            solution = [[0] * n for _ in range(n)]
            
            for i in mincostFlow:
                for j, flow in mincostFlow[i].items():
                    solution[i][j] = flow
            
            return solution
        except Exception as e:
            return [[0] * n for _ in range(n)]