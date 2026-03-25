# solver.py
from typing import Any, Dict, List, Set

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Compute the edge expansion of a directed graph for a given subset S.

        Expansion(S) = |{(u, v) ∈ E | u ∈ S, v ∉ S}| / |S|

        Parameters
        ----------
        problem : dict
            Must contain:
                - "adjacency_list": List[List[int]] adjacency list of G
                - "nodes_S": List[int] nodes in the subset S

        Returns
        -------
        dict
            {"edge_expansion": value}
        """
        adj_list: List[List[int]] = problem.get("adjacency_list", [])
        nodes_S_list: List[int] = problem.get("nodes_S", [])

        n = len(adj_list)
        # Handle trivial cases
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}
        nodes_S_set: Set[int] = set(nodes_S_list)
        if len(nodes_S_set) == n:
            return {"edge_expansion": 0.0}

        # Count edges leaving S
        out_edges = 0
        for u in nodes_S_set:
            # Fast membership test by set
            for v in adj_list[u]:
                if v not in nodes_S_set:
                    out_edges += 1

        expansion = out_edges / len(nodes_S_set)
        return {"edge_expansion": float(expansion)}
