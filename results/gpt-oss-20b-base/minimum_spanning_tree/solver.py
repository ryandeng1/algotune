from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
        using an efficient Kruskal implementation (with path‑compression DSU).
        The resulting edges are sorted by (u, v).

        :param problem: dict with 'num_nodes' and 'edges' (list of [u, v, w])
        :return: dict with 'mst_edges': list of [u, v, w]
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]

        # Disjoint Set Union (Union‑Find) with path compression
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            fx, fy = find(x), find(y)
            if fx == fy:
                return False
            if rank[fx] < rank[fy]:
                parent[fx] = fy
            elif rank[fx] > rank[fy]:
                parent[fy] = fx
            else:
                parent[fy] = fx
                rank[fx] += 1
            return True

        # Sort edges by weight
        edges_sorted = sorted(edges, key=lambda e: e[2])

        mst_edges: List[List[float]] = []
        for u, v, w in edges_sorted:
            if union(u, v):
                # Normalise order of nodes
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])
                if len(mst_edges) == num_nodes - 1:
                    break

        # Final sort by node indices (already sorted by weight, but ensure
        # deterministic order for equal weights)
        mst_edges.sort(key=lambda e: (e[0], e[1]))
        return {"mst_edges": mst_edges}