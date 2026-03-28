import collections
from typing import Dict, List

def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
    """
    Compute the global efficiency of an undirected graph given its adjacency list.
    Global efficiency is defined as the average of the reciprocal of the shortest
    path lengths between all pairs of nodes:
        E = (1 / (n * (n-1))) * sum_{i!=j} 1 / d(i,j)
    If a pair of nodes is disconnected, its contribution to the sum is 0
    (since 1/∞ = 0).

    The implementation uses a fast breadth‑first search from each source node,
    avoiding the overhead of the NetworkX library and all its bookkeeping.
    """
    adj = problem["adjacency_list"]
    n = len(adj)
    # Empty or single node graph: efficiency is 0 by definition
    if n < 2:
        return {"global_efficiency": 0.0}

    # Pre‑allocate a list for distances (integer depths)
    distances = [0] * n
    total_reciprocal = 0.0

    for src in range(n):
        # Reset BFS
        for i in range(n):
            distances[i] = -1
        distances[src] = 0
        q = collections.deque([src])

        while q:
            u = q.popleft()
            d_u = distances[u]
            for v in adj[u]:
                if distances[v] == -1:
                    distances[v] = d_u + 1
                    q.append(v)

        # Sum reciprocals of distances for all j > src to avoid double counting
        for j in range(src + 1, n):
            d = distances[j]
            if d > 0:  # reachable
                total_reciprocal += 1.0 / d

    # Since we summed only for j > src, multiply by 2 to account for both (i,j) and (j,i)
    total_reciprocal *= 2.0
    # Normalization factor: number of ordered pairs of distinct nodes
    denom = n * (n - 1)
    efficiency = total_reciprocal / denom

    return {"global_efficiency": float(efficiency)}