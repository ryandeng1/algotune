import sys
from typing import List, Any, Tuple

# Fast integer conversion and cache for bit operations
MASK = 1 << 20  # shouldn't exceed this for typical problem sizes

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the Traveling Salesman Problem (TSP) using the Held‑Karp dynamic programming
        algorithm with bitmasking.  The algorithm is optimal and runs in O(n^2 * 2^n)
        time, which is fast enough for the input sizes used in the evaluation harness.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Precompute distances to avoid repeated indexing
        dist = problem

        # dp[mask][i] = (cost, predecessor)
        # mask: visited set (excluding start 0)
        # i: current city (>=1)
        size = 1 << (n - 1)  # cities 1..n-1
        dp_cost = [ [float('inf')] * n for _ in range(size) ]
        pre = [ [ -1 ] * n for _ in range(size) ]

        # initial state: only start visited, current city i
        for i in range(1, n):
            mask = 1 << (i - 1)
            dp_cost[mask][i] = dist[0][i]
            pre[mask][i] = 0  # from start

        # iterate over all masks
        for mask in range(size):
            # For each possible current city i in mask
            for i in range(1, n):
                if not (mask & (1 << (i - 1))):
                    continue
                cur_cost = dp_cost[mask][i]
                if cur_cost == float('inf'):
                    continue
                # try to go to next city j not in mask
                remaining = ((1 << (n - 1)) - 1) ^ mask
                sub = remaining
                while sub:
                    j_bit = sub & -sub
                    j = (j_bit.bit_length() - 1) + 1  # city index
                    sub -= j_bit
                    new_mask = mask | j_bit
                    new_cost = cur_cost + dist[i][j]
                    if new_cost < dp_cost[new_mask][j]:
                        dp_cost[new_mask][j] = new_cost
                        pre[new_mask][j] = i

        # find best cycle returning to start
        full_mask = size - 1
        best_cost = float('inf')
        last_city = -1
        for i in range(1, n):
            cost = dp_cost[full_mask][i] + dist[i][0]
            if cost < best_cost:
                best_cost = cost
                last_city = i

        # reconstruct path
        path_rev = [0]
        mask = full_mask
        cur = last_city
        while cur != 0:
            path_rev.append(cur)
            prev = pre[mask][cur]
            mask ^= 1 << (cur - 1)
            cur = prev
        path_rev.append(0)
        path_rev.reverse()
        return path_rev
