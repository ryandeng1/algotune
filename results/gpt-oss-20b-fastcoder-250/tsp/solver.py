def solve(problem: list[list[int]]) -> list[int]:
    """
    Solve the TSP problem using Held‑Karp dynamic programming.
    Works for n <= 15 (2^n * n^2 ≈ 15M operations).
    Returns a tour starting and ending at city 0.
    """
    from functools import lru_cache
    n = len(problem)
    if n <= 1:
        return [0, 0]

    # Pre‑compute neighbours for faster access
    dist = problem
    full_mask = (1 << n) - 1

    @lru_cache(maxsize=None)
    def dp(mask: int, last: int) -> tuple[int, int]:
        """Return (cost, prev_city) for the minimal tour visiting cities in mask ending at last."""
        if mask == (1 << last):
            # Only start city visited
            return 0, -1

        prev_mask = mask ^ (1 << last)
        best_cost = 10**18
        best_prev = -1
        # iterate over all possible previous cities
        sub = prev_mask
        while sub:
            prev_city = sub.bit_length() - 1
            if mask & (1 << prev_city):
                cost, _ = dp(prev_mask, prev_city)
                candidate = cost + dist[prev_city][last]
                if candidate < best_cost:
                    best_cost = candidate
                    best_prev = prev_city
            sub = (sub - 1) & prev_mask
        return best_cost, best_prev

    # Find optimal closing edge from start (0)
    best_total = 10**18
    best_last = -1
    for last in range(1, n):
        cost, _ = dp(full_mask, last)
        candidate = cost + dist[last][0]
        if candidate < best_total:
            best_total = candidate
            best_last = last

    # Reconstruct path
    path = [0] * (n + 1)
    mask = full_mask
    curr = best_last
    pos = n - 1
    while curr != 0:
        path[pos] = curr
        prev_cost, prev_city = dp(mask, curr)
        mask ^= (1 << curr)
        curr = prev_city
        pos -= 1
    path[0] = 0
    path[-1] = 0
    return path