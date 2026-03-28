from typing import Any, Dict, List, Set

class Solver:
    @staticmethod
    def solve(problem: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute the edge expansion of a subset S in a directed graph given by an adjacency list.
        
        The graph is treated as directed: an edge u->v contributes to the boundary
        if u∈S and v∉S (and vice‑versa for reverse edges).
        """
        adj: List[List[int]] = problem["adjacency_list"]
        S_list: List[int] = problem["nodes_S"]
        n: int = len(adj)
        S: Set[int] = set(S_list)

        # Quick exit for empty or full set
        if not S or len(S) == n:
            return {"edge_expansion": 0.0}

        # Count boundary edges
        boundary: int = 0
        for u in range(n):
            in_S = u in S
            for v in adj[u]:
                if (v in S) != in_S:          # exactly one endpoint in S
                    boundary += 1

        denom = min(len(S), n - len(S))
        expansion = boundary / denom if denom else 0.0
        return {"edge_expansion": float(expansion)}