from typing import Any, Dict, List, Set

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute the directed edge expansion of a subset S in a graph
        represented by an adjacency list.

        Edge expansion is defined as

            |δ(S)| / |S|,

        where δ(S) is the set of edges directed from a vertex in S to a
        vertex not in S.  If S is empty or contains all vertices the
        expansion is defined to be 0.0.

        Parameters
        ----------
        problem : dict
            Must contain
            - 'adjacency_list': List[List[int]]
            - 'nodes_S': List[int]

        Returns
        -------
        dict
            {'edge_expansion': <float>}
        """
        adj_list: List[List[int]] = problem["adjacency_list"]
        send_to_S: int = 0
        S: Set[int] = set(problem["nodes_S"])
        n: int = len(adj_list)

        # Edge cases: empty graph or empty S or S == V
        if n == 0 or not S or len(S) == n:
            return {"edge_expansion": 0.0}

        # Count edges from S to V \ S
        for u in S:
            for v in adj_list[u]:
                if v not in S:
                    send_to_S += 1

        expansion = send_to_S / len(S)
        return {"edge_expansion": float(expansion)}