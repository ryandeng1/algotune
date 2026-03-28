from typing import List, Tuple, Dict

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Held‑Karp dynamic programming
        END = 1 << (n - 1)      # mask for node 0
        INF = 10 ** 18

        # dp[mask][i] = minimal cost to start at 0, visit nodes in mask, end at i
        dp: List[Dict[int, int]] = [{} for _ in range(1 << n)]
        parent: List[Dict[int, int]] = [{} for _ in range(1 << n)]

        dp[1 << 0][0] = 0  # start at node 0

        for mask in range(1 << n):
            for u in dp[mask]:
                cost_u = dp[mask][u]
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    new_mask = mask | (1 << v)
                    new_cost = cost_u + problem[u][v]
                    if new_cost < dp[new_mask].get(v, INF):
                        dp[new_mask][v] = new_cost
                        parent[new_mask][v] = u

        full_mask = (1 << n) - 1
        min_cost = INF
        last = -1
        for v in range(1, n):
            c = dp[full_mask].get(v, INF) + problem[v][0]
            if c < min_cost:
                min_cost = c
                last = v

        # reconstruct tour
        tour = [0]
        mask = full_mask
        cur = last
        while cur != 0:
            tour.append(cur)
            prev = parent[mask][cur]
            mask ^= (1 << cur)
            cur = prev
        tour.append(0)
        tour.reverse()
        return tour