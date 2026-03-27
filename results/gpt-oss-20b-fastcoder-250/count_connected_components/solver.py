from typing import Any, Dict


SolutionType = Dict[str, int]


class Solver:
    def solve(self, problem: Dict[str, Any]) -> SolutionType:
        try:
            n = problem.get("num_nodes", 0)
            edges = problem.get("edges", [])
            parent = list(range(n))
            rank = [0] * n

            def find(x: int) -> int:
                # Path compression
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                rx, ry = find(x), find(y)
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
                if 0 <= u < n and 0 <= v < n:
                    union(u, v)

            # Count unique roots
            roots = {find(i) for i in range(n)}
            return {"number_connected_components": len(roots)}
        except Exception:
            return {"number_connected_components": -1}