import math
from typing import List, Tuple, Any

class Solver:
    def solve(self, problem: List[List[float]]) -> List[int]:
        """Solve the Bottleneck Traveling Salesman Problem (BTSP).

        Uses a bit‑mask dynamic program to find a Hamiltonian cycle that
        minimizes the maximum edge weight.  The algorithm runs in
        O(n²·2^n) time and works comfortably for n ≤ 20, which is
        sufficient for the benchmark instances.

        Parameters
        ----------
        problem : List[List[float]]
            Symmetric distance matrix with positive weights.

        Returns
        -------
        List[int]
            A tour starting and ending at city 0 that visits all cities
            exactly once and whose maximum edge weight is minimal.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Pre‑compute edge weights for quick access
        w = problem

        # DP[mask][i] = (best bottleneck to reach i with visited set=mask, prev node)
        # Use sentinel value n for uninitialized
        size = 1 << n
        dp_bottleneck = [[math.inf] * n for _ in range(size)]
        dp_prev = [[-1] * n for _ in range(size)]

        # start at node 0
        dp_bottleneck[1 << 0][0] = 0

        for mask in range(size):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                cur_bott = dp_bottleneck[mask][u]
                if cur_bott == math.inf:
                    continue
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    new_mask = mask | (1 << v)
                    new_bott = max(cur_bott, w[u][v])
                    if new_bott < dp_bottleneck[new_mask][v]:
                        dp_bottleneck[new_mask][v] = new_bott
                        dp_prev[new_mask][v] = u

        full = (1 << n) - 1
        best = math.inf
        best_end = -1
        # close cycle back to 0
        for u in range(1, n):
            bott = max(dp_bottleneck[full][u], w[u][0])
            if bott < best:
                best = bott
                best_end = u

        # reconstruct tour
        path = [0] * (n + 1)
        mask = full
        node = best_end
        for idx in range(n, 0, -1):
            path[idx] = node
            prev = dp_prev[mask][node]
            mask ^= (1 << node)
            node = prev
        path[0] = 0
        path[-1] = 0
        return path
