from typing import Any, List

def solve(problem: dict[str, Any]) -> List[List[Any]]:
    """
    Minimum‑cost perfect assignment using the Hungarian algorithm.
    The input `capacity` matrix only indicates whether an edge can be used (1 = allowed, 0 = forbidden).
    `cost` gives the cost for each allowed edge.
    Returns an `n x n` matrix of flows (0 or 1).
    """
    n = len(problem['capacity'])
    INF = 10 ** 12

    # Build cost matrix with INF for forbidden edges
    cost = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if problem['capacity'][i][j]:
                cost[i][j] = problem['cost'][i][j]

    # Hungarian algorithm (O(n^3))
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] - assigned row for column j
    assigned = [(0, 0)] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assigned[j - 1] = (p[j] - 1, j - 1)

    # Build flow matrix
    flow = [[0] * n for _ in range(n)]
    for i, j in assigned:
        if cost[i][j] < INF:
            flow[i][j] = 1
    return flow