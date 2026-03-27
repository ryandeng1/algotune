from typing import Any, Dict, List

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        rank = self.rank
        parent = self.parent
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
        return True

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of a weighted undirected graph
        using a Kruskal implementation with a custom Union-Find data structure.
        The result is a list of edges [u, v, weight] with u < v, sorted lexicographically.
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]

        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda e: e[2])

        uf = UnionFind(num_nodes)
        mst = []

        for u, v, w in sorted_edges:
            if uf.union(u, v):
                if u > v:
                    u, v = v, u
                mst.append([u, v, w])
                if len(mst) == num_nodes - 1:
                    break

        # The edges are already added in non‑decreasing weight order,
        # but ensure lexicographic order required by the spec.
        mst.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst}