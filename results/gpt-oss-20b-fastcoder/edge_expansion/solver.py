from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        """
        Computes the edge expansion of the supplied subset `S` in the directed graph.

        Edge expansion is defined as:
            |δ(S)| / |S|

        where δ(S) is the set of directed edges with source in `S` and target not in `S`.

        If `S` is empty or contains all nodes, the expansion is defined to be 0.0.
        """

        adj_list = problem['adjacency_list']
        nodes_S_list = problem['nodes_S']

        n = len(adj_list)
        if n == 0:
            return {'edge_expansion': 0.0}

        # Convert to set for O(1) membership tests
        nodes_S = set(nodes_S_list)

        # Edge cases
        if not nodes_S or len(nodes_S) == n:
            return {'edge_expansion': 0.0}

        # Count outgoing edges from S to outside
        edge_count = 0
        for u in nodes_S:
            for v in adj_list[u]:
                if v not in nodes_S:
                    edge_count += 1

        expansion_val = edge_count / len(nodes_S)
        return {'edge_expansion': float(expansion_val)}