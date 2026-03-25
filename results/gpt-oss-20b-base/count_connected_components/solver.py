# solver.py

from typing import Any, Dict, List, Tuple


class Solver:
    """Count connected components in an undirected graph."""

    def _union_find(self, edges: List[Tuple[int, int]], n: int) -> int:
        """
        Compute number of connected components using a
        lightweight Union‑Find (Disjoint Set Union) data structure.
        """
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            # Path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            # Union by rank
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1

        for u, v in edges:
            union(u, v)

        # Count unique roots
        roots = set()
        for i in range(n):
            roots.add(find(i))
        return len(roots)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Count the connected components of an undirected graph.

        Parameters
        ----------
        problem : dict
            Must contain 'num_nodes' (int) and 'edges' (list of tuple[int,int]).

        Returns
        -------
        dict
            {"number_connected_components": int}
        """
        try:
            n = problem.get("num_nodes", 0)
            edges: List[Tuple[int, int]] = problem["edges"]
            # Ensure edges are within bounds; ignore invalid ones
            valid_edges = [(u, v) for u, v in edges if 0 <= u < n and 0 <= v < n]
            cc = self._union_find(valid_edges, n)
            return {"number_connected_components": cc}
        except Exception:
            # Use -1 as an unmistakable “solver errored” sentinel
            return {"number_connected_components": -1}
