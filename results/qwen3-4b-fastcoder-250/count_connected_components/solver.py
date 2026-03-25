from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict:
        n = problem.get("num_nodes", 0)
        edges = problem["edges"]
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            root = x
            while root != parent[root]:
                root = parent[root]
            while x != root:
                nxt = parent[x]
                parent[x] = root
                x = nxt
            return root
        
        def union(x, y):
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
        
        components = 0
        for i in range(n):
            if find(i) == i:
                components += 1
        
        return {"number_connected_components": components}
