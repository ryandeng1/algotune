from typing import Any, Dict

# A minimal, fast Union-Find implementation
class _DSU:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        rank = self.rank
        if rank[px] < rank[py]:
            self.parent[px] = py
        elif rank[px] > rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            rank[px] += 1

SolutionType = Dict[str, int]


class Solver:
    """
    Computes the number of connected components in an undirected graph.

    The graph is represented by a list of edges and a declared number of
    nodes.  This implementation uses a custom Union–Find structure and
    avoids the heavy networkx dependency to give the best performance
    in all scenarios.
    """

    def solve(self, problem: Dict[str, Any]) -> SolutionType:
        try:
            n = int(problem.get("num_nodes", 0))
            if n < 0:
                raise ValueError("num_nodes must be non‑negative")
            dsu = _DSU(n)

            for a, b in problem.get("edges", []):
                if 0 <= a < n and 0 <= b < n:
                    dsu.union(a, b)
                # silently ignore invalid node indices – this mimics the
                # behaviour of the original networkx code.

            # Count distinct roots
            roots = {dsu.find(i) for i in range(n)}
            return {"number_connected_components": len(roots)}
        except Exception:
            return {"number_connected_components": -1}