from typing import List

def solve(problem: List[List[int]]) -> List[int]:
    """
    Solve the TSP problem with dynamic programming (Held‑Karp) in O(n^2 · 2ⁿ).
    Works for n ≤ 20 comfortably and returns the optimal tour starting and
    ending at city 0.
    """
    n = len(problem)
    if n <= 1:
        return [0, 0]

    # Pre‑allocate DP and predecessor arrays
    NSTATE = 1 << n
    INF = 10 ** 18
    dp = [[INF] * n for _ in range(NSTATE)]
    prev = [[-1] * n for _ in range(NSTATE)]

    # Start from city 0, mask 1<<0
    dp[1 << 0][0] = 0

    # Iterate over all subsets that include city 0
    for mask in range(NSTATE):
        if not (mask & 1):
            continue
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            cur = dp[mask][u]
            if cur == INF:
                continue
            # Try to go to next city v not in mask
            not_mask = (~mask) & (NSTATE - 1)
            v = not_mask
            while v:
                lsb = v & -v
                v -= lsb
                j = (lsb.bit_length() - 1)
                new_mask = mask | lsb
                cost = cur + problem[u][j]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    prev[new_mask][j] = u

    # Finish tour: return to city 0
    full_mask = (1 << n) - 1
    best = INF
    last = -1
    for u in range(1, n):
        cost = dp[full_mask][u] + problem[u][0]
        if cost < best:
            best = cost
            last = u

    # Reconstruct path
    path = [0] * (n + 1)
    mask = full_mask
    cur = last
    for i in range(n, 0, -1):
        path[i] = cur
        cur = prev[mask][cur]
        mask ^= (1 << cur)
    path[0] = 0
    path.append(0)
    return path