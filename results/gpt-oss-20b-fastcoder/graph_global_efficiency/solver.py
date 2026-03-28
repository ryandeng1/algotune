from collections import deque
from typing import Dict, List

def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
    """
    Computes the global efficiency of an undirected graph without using NetworkX.
    Global efficiency = (1 / (n*(n-1))) * sum_{i!=j} 1/d(i,j) where d(i,j) is the shortest
    path length between i and j.  Pairs with no path are ignored in the sum.

    The adjacency list is 0-indexed.
    """
    adj = problem['adjacency_list']
    n = len(adj)
    if n <= 1:
        return {'global_efficiency': 0.0}

    # Preprocess adjacency lists to ensure each neighbor set is a list (already is)
    # Prepare a list of neighbors for faster iteration
    neighbors = adj  # alias

    total_inv = 0.0  # sum of 1/d(i,j)

    # For each source node, run BFS to compute distances
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            d_u = dist[u] + 1
            for w in neighbors[u]:
                if dist[w] == -1:
                    dist[w] = d_u
                    q.append(w)
        # accumulate 1/d for all j > s to avoid double counting
        for t in range(s + 1, n):
            d = dist[t]
            if d > 0:
                total_inv += 2.0 / d  # contribute both (s,t) and (t,s)

    denom = n * (n - 1)
    efficiency = total_inv / denom if denom else 0.0
    return {'global_efficiency': float(efficiency)}