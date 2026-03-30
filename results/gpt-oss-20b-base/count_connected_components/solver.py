from typing import Any, Dict

SolutionType = Dict[str, int]


class Solver:
    """
    Optimised solver that counts the number of connected components in an undirected graph
    using a simple disjoint‑set (Union–Find) data structure.  
    This avoids the heavy overhead of :mod:`networkx` and runs in near linear time.
    """

    def _find(self, parent: list[int], x: int) -> int:
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(self, parent: list[int], rank: list[int], a: int, b: int) -> bool:
        ra = self._find(parent, a)
        rb = self._find(parent, b)
        if ra == rb:
            return False
        # Union by rank
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    def solve(self, problem: dict[str, Any]) -> SolutionType:
        try:
            n = problem.get("num_nodes", 0)
            if n <= 0:
                return {"number_connected_components": 0}

            edges = problem.get("edges", [])
            # Initialise DSU structures
            parent = list(range(n))
            rank = [0] * n
            comps = n

            for a, b in edges:
                if a < 0 or a >= n or b < 0 or b >= n:
                    # Skip invalid edges; keep components count unchanged
                    continue
                if self._union(parent, rank, a, b):
                    comps -= 1

            return {"number_connected_components": comps}
        except Exception:
            return {"number_connected_components": -1}