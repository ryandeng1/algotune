from __future__ import annotations
from typing import Any, List, Tuple, Dict

class Solver:
    """Fast MST solver without networkx."""

    class _DSU:
        __slots__ = ("parent", "rank")

        def __init__(self, n: int):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x: int) -> int:
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, a: int, b: int) -> bool:
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return False
            if self.rank[ra] < self.rank[rb]:
                self.parent[ra] = rb
            elif self.rank[ra] > self.rank[rb]:
                self.parent[rb] = ra
            else:
                self.parent[rb] = ra
                self.rank[ra] += 1
            return True

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph.
        The input graph is given as:
          * `num_nodes` – number of nodes, 0‑based indexing
          * `edges` – list of (u, v, w) with 0‑based node indices and float weight

        Returns a dictionary with key 'mst_edges' containing a sorted list of
        edges in the MST as [u, v, w] with u <= v.
        """
        n: int = problem["num_nodes"]
        edges: List[Tuple[int, int, float]] = problem["edges"]

        # Sort edges by weight once – Kruskal’s algorithm
        edges.sort(key=lambda e: e[2])

        dsu = self._DSU(n)
        mst: List[List[float]] = []

        for u, v, w in edges:
            if dsu.union(u, v):
                if u > v:
                    u, v = v, u
                mst.append([u, v, w])
                if len(mst) == n - 1:
                    break

        # Final sort by node indices to meet the required ordering
        mst.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst}