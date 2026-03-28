from typing import Any, Dict, List, Set

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute the edge expansion of the subset S in the given directed graph.
        Edge expansion = |∂S| / |S| where ∂S is the set of directed edges
        that leave S. If S is empty or contains all vertices, the expansion
        is defined as 0.0.
        """
        adj: List[List[int]] = problem["adjacency_list"]
        S_list: List[int] = problem["nodes_S"]

        n: int = len(adj)
        if n == 0 or not S_list:
            return {"edge_expansion": 0.0}

        S: Set[int] = set(S_list)
        if len(S) == n:
            return {"edge_expansion": 0.0}

        # Count edges from S to outside S
        cut_edges: int = 0
        for u in S:
            for v in adj[u]:
                if v not in S:
                    cut_edges += 1

        expansion = cut_edges / len(S)
        return {"edge_expansion": float(expansion)}