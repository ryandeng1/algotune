import networkx as nx

class Solver:
    def solve(self, problem: tuple[dict[str, dict[str, float]], int]) -> list[str]:
        G_dict, k = problem
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)
        
        nodes = list(graph.nodes)
        n = len(nodes)
        dist_matrix = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist_matrix[i][i] = 0
        
        for i, node in enumerate(nodes):
            for neighbor, weight in graph[node].items():
                j = nodes.index(neighbor)
                dist_matrix[i][j] = weight
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i][k] + dist_matrix[k][j] < dist_matrix[i][j]:
                        dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
        
        remaining_nodes = set(nodes)
        centers = []
        for _ in range(k):
            candidate = None
            max_dist = -1
            for node in remaining_nodes:
                current_max = -1
                node_idx = nodes.index(node)
                for other in remaining_nodes:
                    if other == node:
                        continue
                    other_idx = nodes.index(other)
                    d = dist_matrix[node_idx][other_idx]
                    if d > current_max:
                        current_max = d
                if current_max > max_dist:
                    max_dist = current_max
                    candidate = node
            if candidate is None:
                break
            remaining_nodes.remove(candidate)
            centers.append(candidate)
        return centers
