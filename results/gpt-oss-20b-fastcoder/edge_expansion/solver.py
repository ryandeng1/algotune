# solver.py
from __future__ import annotations
from typing import Any, Dict, List, Set

class Solver:
    """
    Calculates the directed edge expansion of a subset of nodes in a graph.
    Edge expansion is defined as the number of directed edges leaving the
    subset divided by the size of the subset.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """
        Parameters
        ----------
        problem : dict
            Must contain:
                - "adjacency_list": list[list[int]] adjacency list of a directed graph
                - "nodes_S": list[int] subset of node indices

        Returns
        -------
        dict
            {"edge_expansion": float}
        """
        adj_list: List[List[int]] = problem.get("adjacency_list", [])
        nodes_S: Set[int] = set(problem.get("nodes_S", []))

        n = len(adj_list)
        if n == 0 or not nodes_S or len(nodes_S) == n:
            return {"edge_expansion": 0.0}

        # Count outgoing edges from S to its complement
        boundary_edges = 0
        comp = set(range(n)) - nodes_S
        for u in nodes_S:
            # iterate only if u is a valid index
            if 0 <= u < n:
                for v in adj_list[u]:
                    if v in comp:
                        boundary_edges += 1

        # Edge expansion: |∂S| / |S|
        expansion = boundary_edges / len(nodes_S)
        return {"edge_expansion": float(expansion)}