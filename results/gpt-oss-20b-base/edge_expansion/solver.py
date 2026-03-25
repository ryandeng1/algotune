# solver.py
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        adj_list: List[List[int]] = problem.get("adjacency_list", [])
        nodes_S_list: List[int] = problem.get("nodes_S", [])

        n = len(adj_list)
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}

        nodes_S = set(nodes_S_list)
        if len(nodes_S) == n:
            return {"edge_expansion": 0.0}

        # Count edges from S to V\S
        out_count = 0
        for u in nodes_S:
            for v in adj_list[u]:
                if v not in nodes_S:
                    out_count += 1

        expansion = out_count / len(nodes_S)
        return {"edge_expansion": expansion}
