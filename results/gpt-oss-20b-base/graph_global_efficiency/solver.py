def solve(problem: dict[str, list[list[int]]]) -> dict[str, float]:
    """
    Computes the global efficiency of an unweighted undirected graph
    given as an adjacency list.  The global efficiency is the average
    value of 1/d(i,j) over all unordered pairs of distinct nodes
    (where d(i,j) is the shortest path length and d(i,j)=∞ if the
    nodes are disconnected).  The implementation uses a breadth
    first search from every node, which is efficient for sparse
    graphs and avoids importing NetworkX.

    Args:
        problem: {"adjacency_list": [[...], ...]}

    Returns:
        {"global_efficiency": value}
    """
    adj = problem.get("adjacency_list", [])
    n = len(adj)
    if n <= 1:
        return {"global_efficiency": 0.0}

    # Build adjacency list with zero‑based indices
    # Ensure the graph is undirected (add missing edges)
    G = [[] for _ in range(n)]
    for u, neigh in enumerate(adj):
        for v in neigh:
            if 0 <= v < n:
                G[u].append(v)
                G[v].append(u)

    total = 0.0
    pairs = n * (n - 1)  # unordered pairs count

    # Preallocate distance list
    from collections import deque
    dist = [-1] * n

    for src in range(n):
        # Reset distances
        for i in range(n):
            dist[i] = -1
        dist[src] = 0
        q = deque([src])

        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in G[u]:
                if dist[v] == -1:
                    dist[v] = du
                    q.append(v)

        # Accumulate contributions of reachable nodes
        for target in range(src + 1, n):
            d = dist[target]
            if d > 0:          # reachable and distinct
                total += 1.0 / d
            # unreachable pairs contribute 0

    # Average over all unordered pairs
    efficiency = total / pairs
    return {"global_efficiency": float(efficiency)}