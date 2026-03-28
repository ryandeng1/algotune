from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        """
        Calculates the edge expansion of a subset S in a directed graph.
        Edge expansion is defined as: |{(u, v) | u ∈ S, v ∉ S}| / |S|

        Parameters
        ----------
        problem : dict
            Must contain:
            - "adjacency_list": List[List[int]] of outgoing neighbors for each node.
            - "nodes_S": List[int] of nodes belonging to the subset S.

        Returns
        -------
        dict[str, float]
            {"edge_expansion": expansion_value}
        """
        adj_list = problem['adjacency_list']
        nodes_S_set = set(problem['nodes_S'])
        n = len(adj_list)

        # Edge cases: empty set or all nodes
        if not nodes_S_set or len(nodes_S_set) == n:
            return {"edge_expansion": 0.0}

        # Count edges leaving S
        boundary = 0
        for u in nodes_S_set:
            for v in adj_list[u]:
                if v not in nodes_S_set:
                    boundary += 1

        expansion = boundary / len(nodes_S_set)
        return {"edge_expansion": float(expansion)}