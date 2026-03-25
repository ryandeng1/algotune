import heapq

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        adj = [[] for _ in range(num_nodes)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        visited = [False] * num_nodes
        heap = []
        heapq.heappush(heap, (0.0, 0, -1))
        
        mst_edges = []
        
        while heap:
            weight, node, parent = heapq.heappop(heap)
            if visited[node]:
                continue
            visited[node] = True
            if parent != -1:
                mst_edges.append([parent, node, weight])
            for neighbor, w in adj[node]:
                if not visited[neighbor]:
                    heapq.heappush(heap, (w, neighbor, node))
        
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst_edges}
