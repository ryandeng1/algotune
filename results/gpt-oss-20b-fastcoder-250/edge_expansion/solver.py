from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        """
        Calculates the edge expansion for the given subset S in the graph.

        Edge expansion is defined as the number of directed edges leaving the set
        S (i.e., from S to V\\S) divided by the size of S.
        """
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        nodes_S = set(nodes_S_list)

        # Edge cases: empty graph, empty S, or S contains all nodes
        if n == 0 or not nodes_S or len(nodes_S) == n:
            return {"edge_expansion": 0.0}

        # Count directed edges from S to V\\S
        out_edges = 0
        for u in nodes_S:
            # Iterate through adjacency list of u
            for v in adj_list[u]:
                if v not in nodes_S:  # edge leaves S
                    out_edges += 1

        expansion_value = out_edges / len(nodes_S)
        return {"edge_expansion": expansion_value}