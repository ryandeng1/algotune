# solver.py
import itertools
from collections import defaultdict
import math

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Bottleneck Traveling Salesman Problem (BTSP) using a dynamic
        programming approach similar to Held‑Karp.  The DP state keeps track of
        the minimal possible bottleneck (maximum edge weight) for reaching a
        subset of cities ending at a particular city.

        Parameters
        ----------
        problem : list[list[float]]
            Symmetric distance matrix (n x n) with positive weights.

        Returns
        -------
        list[int]
            Optimal tour starting and ending at city 0 (length n+1).
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Convert to 0/1 indexed np array for faster access
        import numpy as np
        dist = np.array(problem, dtype=float)

        INF = 1e100
        full_mask = (1 << n) - 1

        # dp[mask][j] = minimal bottleneck to reach city j having visited set mask (including j)
        dp = [defaultdict(lambda: INF) for _ in range(1 << n)]
        parent = [dict() for _ in range(1 << n)]

        # start from city 0
        dp[1][0] = 0

        for mask in range(1, full_mask + 1):
            # skip masks that don't contain start city 0
            if not (mask & 1):
                continue
            for last in list(dp[mask].keys()):
                cur_bottleneck = dp[mask][last]
                # try to go to a new city
                for nxt in range(n):
                    nxt_bit = 1 << nxt
                    if mask & nxt_bit:
                        continue
                    new_mask = mask | nxt_bit
                    new_bott = max(cur_bottleneck, dist[last, nxt])
                    if new_bott < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = new_bott
                        parent[new_mask][nxt] = last

        # Find best end city to return to 0
        best_bott = INF
        best_end = -1
        for last in range(1, n):
            bott = max(dp[full_mask][last], dist[last, 0])
            if bott < best_bott:
                best_bott = bott
                best_end = last

        if best_end == -1:
            return []

        # Reconstruct path
        path = [0] * (n + 1)
        path[0] = 0
        mask = full_mask
        cur = best_end
        for i in range(n, 0, -1):
            path[i] = cur
            prev = parent[mask][cur]
            mask ^= (1 << cur)
            cur = prev
        path[n] = 0
        return path
