from typing import Any, Dict, Iterable

SolutionType = Dict[str, int]

class Solver:
    def _union_find(self, n: int, edges: Iterable[tuple[int, int]]) -> int:
        parent = list(range(n))
        rank = [0] * n
        def find(x: int) -> int:
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

        seen_roots = {find(i) for i in range(n)}
        return len(seen_roots)

    def solve(self, problem: Dict[str, Any]) -> SolutionType:
        try:
            n = problem.get('num_nodes', 0)
            edges = problem.get('edges', [])
            if not isinstance(edges, list):
                return {'number_connected_components': -1}
            cc = self._union_find(n, edges)
            return {'number_connected_components': cc}
        except Exception:
            return {'number_connected_components': -1}