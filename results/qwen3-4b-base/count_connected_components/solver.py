from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, int]:
        try:
            n = problem.get("num_nodes", 0)
            edges = problem.get("edges", [])
            if n == 0:
                return {"number_connected_components": 0}
            parent = list(range(n))
            rank = [0] * n

            def find(x: int) -> int:
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x: int, y: int):
                rx = find(x)
                ry = find(y)
                if rx == ry:
                    return
                if rank[rx] < rank[ry]:
                    parent[rx] = ry
                elif rank[rx] > rank[ry]:
                    parent[ry] = rx
                else:
                    parent[ry] = rx
                    rank[rx] += 1

            for u, v in edges:
                union(u, v)

            roots = set(find(i) for i in range(n))
            return {"number_connected_components": len(roots)}
        except Exception:
            return {"number_connected_components": -1}