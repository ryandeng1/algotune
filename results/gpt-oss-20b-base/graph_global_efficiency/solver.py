from collections import deque
from typing import Dict, List

def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
    """
    Calculates the global efficiency of the graph using a fast BFS approach.
    Global efficiency is the average of reciprocal of shortest path lengths
    between every pair of distinct nodes.

    Args:
        problem: A dictionary containing the adjacency list of the graph.
                 {"adjacency_list": adj_list}

    Returns:
        A dictionary containing the global efficiency.
        {"global_efficiency": efficiency_value}
    """
    adj = problem["adjacency_list"]
    n = len(adj)

    # Edge cases: efficiency is 0 when there is 0 or 1 node
    if n <= 1:
        return {"global_efficiency": 0.0}

    # Pre‑allocate result accumulator
    total_reciprocal = 0.0
    # All paths to all other nodes (exclude self)
    pairs = n * (n - 1)

    # For each source node, run BFS to get distances
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du
                    q.append(v)

        # Sum reciprocal distances to other nodes
        # Distances of -1 (unreachable) are treated as infinite -> 0 contribution
        for tgt in range(n):
            if tgt == src:
                continue
            d = dist[tgt]
            if d > 0:
                total_reciprocal += 1.0 / d
            # unreachable nodes contribute 0 directly

    global_efficiency = total_reciprocal / pairs
    return {"global_efficiency": float(global_efficiency)}