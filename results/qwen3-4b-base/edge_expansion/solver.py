from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        nodes_S = set(nodes_S_list)

        if n == 0 or not nodes_S:
            return {"edge_expansion": 0.0}
        if len(nodes_S) == n:
            return {"edge_expansion": 0.0}

        total_edges = 0
        for u in nodes_S:
            for v in adj_list[u]:
                if v not in nodes_S:
                    total_edges += 1

        return {"edge_expansion": total_edges / len(nodes_S)}