# solver.py
import sys
from typing import Any, Dict, List, Tuple

# Union-Find (Disjoint Set Union) implementation
class _DSU:
    __slots__ = ("parent", "rank")
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
        using Kruskal's algorithm.

        Parameters
        ----------
        problem : dict
            Must contain:
            * "num_nodes" : int
            * "edges" : List[List[float]] where each element is [u, v, weight]

        Returns
        -------
        dict
            {"mst_edges": List[List[float]]} sorted by (u, v)
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]  # list of [u, v, weight]

        # Sort edges by weight ascending; stable sort keeps consistent order for equal weights
        sorted_edges: List[Tuple[int, int, float]] = sorted(
            ((int(u), int(v), float(w)) for u, v, w in edges),
            key=lambda e: e[2]
        )

        dsu = _DSU(num_nodes)
        mst: List[List[float]] = []

        # Kruskal's algorithm
        for u, v, w in sorted_edges:
            if dsu.union(u, v):
                # Ensure u < v for consistent sorting later
                if u > v:
                    u, v = v, u
                mst.append([u, v, w])
                if len(mst) == num_nodes - 1:
                    break

        # Final sort by (u, v)
        mst.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst}
