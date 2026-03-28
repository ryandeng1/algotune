from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        if n <= 1:
            return {"global_efficiency": 0.0}

        G = nx.Graph({i: adj_list[i] for i in range(n)})

        try:
            efficiency = nx.global_efficiency(G)
        except Exception:
            return {"global_efficiency": 0.0}

        return {"global_efficiency": float(efficiency)}