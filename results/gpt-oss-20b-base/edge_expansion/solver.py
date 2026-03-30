from __future__ import annotations
from typing import Any, Dict

class Solver:
    """
    Efficient edge–expansion calculator for directed graphs.

    The implementation bypasses NetworkX entirely and operates directly on the
    adjacency list supplied in ``problem['adjacency_list']``.  For a subset
    ``S`` it counts the number of edges going from ``S`` to its complement
    (``∂S``) and divides by ``min(|S|, |V \ S|)``.  This is equivalent to
    NetworkX's ``edge_expansion`` for directed graphs, but avoids the
    heavyweight graph construction and the safety checks performed by
    NetworkX.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        adj_list = problem.get("adjacency_list", [])
        nodes_S_list = problem.get("nodes_S", [])

        n = len(adj_list)
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}

        # Convert the list of nodes in S to a set for O(1) membership tests
        nodes_S = set(nodes_S_list)
        sz_S = len(nodes_S)
        if sz_S == n:
            return {"edge_expansion": 0.0}

        # Count edges leaving S
        cut_edges = 0
        for u in nodes_S:
            # ``adj_list`` indices correspond to node ids; skip missing indices
            if u < 0 or u >= n:  # defensive check
                continue
            neighbors = adj_list[u]
            # Count how many neighbors are outside S
            for v in neighbors:
                if v not in nodes_S:
                    cut_edges += 1

        # Edge expansion for directed graphs is |∂S| / min(|S|, |V\S|)
        denom = sz_S if sz_S <= n - sz_S else n - sz_S
        expansion_value = cut_edges / denom if denom else 0.0

        return {"edge_expansion": expansion_value}