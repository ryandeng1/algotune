from typing import Any

SolutionType = dict[str, int]

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
            
            for a, b in edges:
                if 0 <= a < n and 0 <= b < n:
                    union(a, b)
            
            # Count unique roots
            roots = {find(i) for i in range(n)}
            return {"number_connected_components": len(roots)}
        except Exception:
            return {"number_connected_components": -1}