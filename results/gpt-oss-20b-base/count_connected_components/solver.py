from typing import Any, Dict

SolutionType = Dict[str, int]

class Solver:
    def solve(self, problem: Dict[str, Any]) -> SolutionType:
        # Union‑Find (Disjoint Set Union) without path compression overhead.
        n = problem.get('num_nodes', 0)
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
            else:
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1

        for u, v in problem.get('edges', []):
            if 0 <= u < n and 0 <= v < n:
                union(u, v)

        # Count distinct roots
        roots = set()
        for i in range(n):
            roots.add(find(i))
        return {'number_connected_components': len(roots)}