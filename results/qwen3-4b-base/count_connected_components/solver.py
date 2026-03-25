from collections import deque

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict:
        n = problem["num_nodes"]
        adj = [[] for _ in range(n)]
        for u, v in problem["edges"]:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = [False] * n
        count = 0
        
        for i in range(n):
            if not visited[i]:
                count += 1
                queue = deque([i])
                visited[i] = True
                while queue:
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
        
        return {"number_connected_components": count}
