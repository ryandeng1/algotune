# solver.py
import math
from functools import lru_cache
from typing import List, Any

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the Traveling Salesman Problem (TSP) for a small number of cities
        using the Held-Karp dynamic programming algorithm (bitmask DP).
        The function returns an optimal tour that starts and ends at city 0.

        Parameters
        ----------
        problem : List[List[int]]
            Adjacency matrix with positive integer distances.  `problem[i][j]`
            gives the distance from city `i` to city `j`.  The matrix is
            assumed to be symmetric and have zeros on the diagonal.

        Returns
        -------
        List[int]
            A list of city indices representing the optimal tour, starting
            and ending at city 0.  The length of the list is `n + 1` where
            `n` is the number of cities.  If the problem has only one city,
            the result is `[0, 0]`.  If for any reason the algorithm fails,
            an empty list is returned.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Pre‑compute distances for speed
        dist = problem
        FULL_MASK = (1 << n) - 1

        # dp[mask][i] -> minimal cost to start at 0, visit cities in
        # `mask` and end at city `i`.  `mask` always includes city 0.
        dp = [dict() for _ in range(1 << n)]
        # base case: only 0 visited, we are at 0 with cost 0
        dp[1 << 0][0] = 0

        # iterate over all masks that include city 0
        for mask in range(1 << n):
            if not (mask & 1):
                continue  # skip masks that do not include city 0
            for last in dp[mask]:
                cur_cost = dp[mask][last]
                # try to go to a new city
                not_visited = (~mask) & FULL_MASK
                while not_visited:
                    nxt = (not_visited & -not_visited).bit_length() - 1
                    not_visited &= not_visited - 1
                    new_mask = mask | (1 << nxt)
                    new_cost = cur_cost + dist[last][nxt]
                    if nxt not in dp[new_mask] or new_cost < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = new_cost

        # Find best last city to return to 0
        best_cost = math.inf
        best_last = None
        for last, cost in dp[FULL_MASK].items():
            total = cost + dist[last][0]
            if total < best_cost:
                best_cost = total
                best_last = last

        if best_last is None:
            return []

        # Reconstruct path
        path = [0] * (n + 1)
        mask = FULL_MASK
        curr = best_last
        for pos in range(n, 0, -1):
            path[pos] = curr
            prev_cost = dp[mask][curr]
            # find previous city that leads to current with optimal cost
            mask ^= (1 << curr)  # remove current
            for prev, pc in dp[mask].items():
                if pc + dist[prev][curr] == prev_cost:
                    curr = prev
                    break
        path[0] = 0
        return path
