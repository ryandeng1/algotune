# solver.py
import sys
from typing import Any, Dict, List, Tuple

# Union‑Find data structure (path compression + union by size)
class _UF:
    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        # union by size
        if self.size[pa] < self.size[pb]:
            pa, pb = pb, pa
        self.parent[pb] = pa
        self.size[pa] += self.size[pb]


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, int]:
        """
        Count connected components in an undirected graph.

        Expected `problem` format:
        {
            "num_nodes": int,
            "edges": List[Tuple[int, int]]
        }

        Returns:
        {
            "number_connected_components": int
        }
        """
        try:
            n: int = problem.get("num_nodes", 0)
            if n <= 0:
                return {"number_connected_components": 0}

            uf = _UF(n)

            edges: List[Tuple[int, int]] = problem.get("edges", [])
            for u, v in edges:
                # Assume 0‑based indices; skip invalid indices
                if 0 <= u < n and 0 <= v < n:
                    uf.union(u, v)

            # Count unique roots
            roots = set()
            parent = uf.parent
            for i in range(n):
                # We can use the efficient find via while loop
                x = i
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                roots.add(x)

            return {"number_connected_components": len(roots)}
        except Exception:
            # Signal error
            return {"number_connected_components": -1}
