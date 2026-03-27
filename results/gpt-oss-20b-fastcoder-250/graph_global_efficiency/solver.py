from collections import deque
from typing import Dict, List, Any

def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
    """
    Calculate the global efficiency of an undirected graph given as an adjacency list.
    The adjacency list is a list of lists, where index `i` contains the neighbors of node `i`.
    The algorithm uses breadth‑first search from each vertex (O(n*(n+m))) and avoids
    the heavy NetworkX dependency, providing a significant speed advantage for large graphs.
    """
    adj = problem["adjacency_list"]
    n = len(adj)

    # Edge case: 0 or 1 node
    if n <= 1:
        return {"global_efficiency": 0.0}

    # Pre‑allocate list for BFS distances
    total_inverse_dist = 0.0

    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)

        # Sum 1/d for all distinct nodes j != src
        for j in range(n):
            if j == src:
                continue
            d = dist[j]
            if d > 0:
                total_inverse_dist += 1.0 / d
            # If d == -1 (disconnected), contribution is 0

    # Global efficiency is average over all ordered pairs (i, j), i != j
    efficiency = total_inverse_dist / (n * (n - 1))
    return {"global_efficiency": efficiency}