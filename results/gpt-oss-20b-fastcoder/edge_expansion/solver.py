from typing import Any, Dict, List, Set


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """Compute the edge expansion of a directed graph for the given subset S."""
        adj_list: List[List[int]] = problem["adjacency_list"]
        nodes_s_list: List[int] = problem["nodes_S"]

        n = len(adj_list)
        if n == 0:
            return {"edge_expansion": 0.0}

        nodes_s: Set[int] = set(nodes_s_list)
        if not nodes_s or len(nodes_s) == n:
            return {"edge_expansion": 0.0}

        # Count edges leaving S
        out_edges = 0
        for u in nodes_s:
            for v in adj_list[u]:
                if v not in nodes_s:
                    out_edges += 1

        # Edge expansion: |∂S| / |S|
        expansion = out_edges / len(nodes_s)
        return {"edge_expansion": float(expansion)}