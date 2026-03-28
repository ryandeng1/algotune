from typing import Any, List, Tuple, Dict

class Solver:
    # Union–Find (Disjoint Set Union) implementation
    def _find(self, parent: List[int], x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(self, parent: List[int], rank: List[int], x: int, y: int) -> bool:
        xroot, yroot = self._find(parent, x), self._find(parent, y)
        if xroot == yroot:
            return False
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
        return True

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
        using Kruskal's algorithm with a disjoint-set data structure.

        :param problem: Dictionary with keys:
            - 'num_nodes': int, number of nodes (nodes are numbered 0..num_nodes-1)
            - 'edges': list of tuples (u, v, w) where u, v are node indices and w is a float weight
        :return: Dictionary with key 'mst_edges' containing list of edges [u, v, w] in the MST,
                 sorted lexicographically by (u, v).
        """
        num_nodes = problem["num_nodes"]
        edges: List[Tuple[int, int, float]] = problem["edges"]

        # Sort edges by weight (Kruskal's algorithm)
        edges_sorted = sorted(edges, key=lambda e: e[2])

        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        mst: List[List[float]] = []
        for u, v, w in edges_sorted:
            if self._union(parent, rank, u, v):
                if u > v:
                    u, v = v, u
                mst.append([u, v, w])
                if len(mst) == num_nodes - 1:
                    break

        # Final lexicographic sort by (u, v)
        mst.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst}