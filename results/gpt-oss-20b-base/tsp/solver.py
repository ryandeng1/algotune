from typing import List, Tuple

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the TSP problem using a dynamic‑programming Held‑Karp
        algorithm.  The input is a square distance matrix.  The
        returned tour starts and ends at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # DP table: dp[mask][i] = minimal cost to reach city i,
        # having visited the set of cities indicated by mask
        # (mask includes city 0 always by convention but we shift to
        # keep it simple).
        INF = 10**15
        size = 1 << n
        dp = [[INF] * n for _ in range(size)]
        # start at city 0, cost zero
        dp[1][0] = 0
        # predecessor table for path reconstruction
        prev: List[List[Tuple[int, int]]] = [[(-1, -1)] * n for _ in range(size)]

        for mask in range(1, size):
            # skip masks that don't contain city 0
            if not (mask & 1):
                continue
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                cur_cost = dp[mask][u]
                if cur_cost == INF:
                    continue
                # try to go to an unvisited city v
                remaining = (~mask) & (size - 1)
                v = remaining
                while v:
                    lb = v & -v
                    vi = (lb.bit_length() - 1)
                    next_mask = mask | lb
                    new_cost = cur_cost + problem[u][vi]
                    if new_cost < dp[next_mask][vi]:
                        dp[next_mask][vi] = new_cost
                        prev[next_mask][vi] = (mask, u)
                    v -= lb

        # Close the tour by returning to city 0
        full_mask = size - 1
        best_cost = INF
        last_city = -1
        for i in range(1, n):
            cost = dp[full_mask][i] + problem[i][0]
            if cost < best_cost:
                best_cost = cost
                last_city = i

        # Reconstruct path
        path = [0]
        mask = full_mask
        cur = last_city
        while cur != 0:
            path.append(cur)
            mask, cur = prev[mask][cur]
        path.append(0)
        path.reverse()
        return path