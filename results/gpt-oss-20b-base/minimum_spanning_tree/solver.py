from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        # Kruskal's algorithm with a disjoint‑set structure
        edges = problem["edges"]
        # Ensure each edge is stored as (weight, u, v) for easy sorting
        sorted_edges = sorted([(w, u, v) for u, v, w in edges])

        parent = list(range(problem["num_nodes"]))
        rank = [0] * problem["num_nodes"]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        mst_edges: List[List[float]] = []
        for w, u, v in sorted_edges:
            if union(u, v):
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])

        return {"mst_edges": mst_edges}