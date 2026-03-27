from typing import Any, Dict, List, Set


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute the edge expansion of subset S in a directed graph given by an adjacency list.

        Parameters
        ----------
        problem : dict
            Must contain keys "adjacency_list" (List[List[int]]) and "nodes_S" (List[int]).

        Returns
        -------
        dict
            {"edge_expansion": float}
        """
        adj_list: List[List[int]] = problem["adjacency_list"]
        nodes_S_list: List[int] = problem["nodes_S"]

        n = len(adj_list)
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}

        nodes_S: Set[int] = set(nodes_S_list)
        if len(nodes_S) == n:
            return {"edge_expansion": 0.0}

        # Sum edges from S to the complement
        border = 0
        for u in nodes_S:
            for v in adj_list[u]:
                if v not in nodes_S:
                    border += 1

        expansion = border / len(nodes_S)
        return {"edge_expansion": float(expansion)}