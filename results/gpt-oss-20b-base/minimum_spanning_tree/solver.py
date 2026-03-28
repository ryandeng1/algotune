from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Construct MST with Kruskal's algorithm and return edges sorted by (u, v).
        """
        edges = problem["edges"]
        num_nodes = problem["num_nodes"]

        # Sort edges by weight
        edges_sorted = sorted(edges, key=lambda e: e[2])

        # Union-Find data structure
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
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

        mst_edges = []
        for u, v, w in edges_sorted:
            if union(u, v):
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])
                if len(mst_edges) == num_nodes - 1:
                    break

        mst_edges.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst_edges}