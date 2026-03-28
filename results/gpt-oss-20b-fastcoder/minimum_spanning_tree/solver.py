from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
        using Kruskal's algorithm and a union‑find data structure.
        The returned edges are sorted by (u, v) in ascending order.

        :param problem: dict with keys 'num_nodes' and 'edges'
        :return: dict with key 'mst_edges' containing a list of [u, v, weight]
        """
        # ---------- Union‑Find ----------
        parent = list(range(problem['num_nodes']))
        rank = [0] * problem['num_nodes']

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            xroot, yroot = find(x), find(y)
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

        # ---------- Kruskal ----------
        edges = problem['edges']
        # sort edges by weight
        edges.sort(key=lambda e: e[2])

        mst = []
        for u, v, w in edges:
            if union(u, v):
                if u > v:
                    u, v = v, u
                mst.append([u, v, w])
                if len(mst) == problem['num_nodes'] - 1:
                    break

        mst.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst}