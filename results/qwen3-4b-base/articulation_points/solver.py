class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        graph = [[] for _ in range(num_nodes)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        disc = [-1] * num_nodes
        low = [-1] * num_nodes
        parent = [-1] * num_nodes
        time = [0]
        ap = set()
        
        def dfs(u):
            disc[u] = time[0]
            low[u] = time[0]
            time[0] += 1
            children = 0
            for v in graph[u]:
                if disc[v] == -1:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if parent[u] == -1 and children > 1:
                        ap.add(u)
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        
        for i in range(num_nodes):
            if disc[i] == -1:
                dfs(i)
        
        return {"articulation_points": sorted(ap)}
