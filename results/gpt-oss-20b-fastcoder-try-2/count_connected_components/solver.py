from typing import Any, Dict

SolutionType = Dict[str, int]

class Solver:
    def solve(self, problem: Dict[str, Any]) -> SolutionType:
        try:
            n = problem.get('num_nodes', 0)
            edges = problem.get('edges', [])
            parent = list(range(n))
            rank = [0]*n

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                xr, yr = find(x), find(y)
                if xr == yr:
                    return
                if rank[xr] < rank[yr]:
                    parent[xr] = yr
                elif rank[xr] > rank[yr]:
                    parent[yr] = xr
                else:
                    parent[yr] = xr
                    rank[xr] += 1

            for u, v in edges:
                if 0 <= u < n and 0 <= v < n:
                    union(u, v)

            comps = len({find(i) for i in range(n)})
            return {'number_connected_components': comps}
        except Exception:
            return {'number_connected_components': -1}