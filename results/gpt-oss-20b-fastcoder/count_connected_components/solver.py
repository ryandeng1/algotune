from typing import Any, Dict

SolutionType = Dict[str, int]


class Solver:
    def solve(self, problem: dict[str, Any]) -> SolutionType:
        try:
            n = problem.get("num_nodes", 0)
            edges = problem.get("edges", [])

            parent = list(range(n))
            rank = [0] * n

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[ra] > rank[rb]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1

            for u, v in edges:
                if 0 <= u < n and 0 <= v < n:
                    union(u, v)

            # Count unique roots
            components = set()
            for i in range(n):
                components.add(find(i))
            return {"number_connected_components": len(components)}
        except Exception:
            return {"number_connected_components": -1}