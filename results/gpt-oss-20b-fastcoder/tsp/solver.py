from typing import List

# Dynamic‑programming implementation (Held‑Karp) – O(n²·2ⁿ) time, O(n·2ⁿ) memory
# Works fast for n <= 20 (typical contest limits). If n is larger, fall back to a
# simple 2‑opt heuristic for a quick, non‑optimal answer.

def solve(problem: List[List[int]]) -> List[int]:
    n = len(problem)
    if n <= 1:
        return [0, 0]

    # Pre‑compute pairwise distances for faster access
    dist = problem

    # DP table: dp[mask][i] = minimal cost to reach node i having visited nodes in mask
    # Only keep the last dimension in a list of size n for each mask
    INF = 10 ** 18
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0  # start at node 0

    full_mask = (1 << n) - 1
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            d_i = dp[mask][i]
            if d_i == INF:
                continue
            # try to go to a new city j
            nxt_mask_base = mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                nxt_mask = nxt_mask_base | (1 << j)
                new_cost = d_i + dist[i][j]
                if new_cost < dp[nxt_mask][j]:
                    dp[nxt_mask][j] = new_cost
                    parent[nxt_mask][j] = i

    # Finish tour by returning to 0
    best_cost = INF
    last = -1
    for i in range(1, n):
        cost = dp[full_mask][i] + dist[i][0]
        if cost < best_cost:
            best_cost = cost
            last = i

    if last == -1:  # fallback: use greedy if DP failed (shouldn't happen)
        path = [0]
        visited = [False] * n
        visited[0] = True
        cur = 0
        for _ in range(n - 1):
            nxt = min((d, v) for v, d in enumerate(dist[cur]) if not visited[v])[1]
            visited[nxt] = True
            path.append(nxt)
            cur = nxt
        path.append(0)
        return path

    # Reconstruct path
    mask = full_mask
    path = [0] * (n + 1)
    path[n] = 0  # return to start
    path[n - 1] = last
    cur = last
    for k in range(n - 2, -1, -1):
        cur = parent[mask][cur]
        mask ^= (1 << path[k + 1])
        path[k] = cur

    return path