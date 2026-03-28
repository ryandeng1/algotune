from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        edges = [(u, v) for u in range(n) for v in adj_list[u] if u < v]
        
        if not edges:
            return {"global_efficiency": 0.0}
        
        G = nx.Graph(edges)
        
        try:
            efficiency = nx.global_efficiency(G)
        except Exception as e:
            return {"global_efficiency": 0.0}
        
        return {"global_efficiency": float(efficiency)}