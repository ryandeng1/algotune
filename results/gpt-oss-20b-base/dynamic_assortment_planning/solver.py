# solver.py
from typing import Any, List


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the Dynamic Assortment Planning problem using a maximum weight
        bipartite matching approach.  The problem is transformed into an
        assignment problem where each period is a left node and each
        capacity unit of every product is a right node.  An additional set
        of dummy nodes represents the choice to stay idle.  The Hungarian
        algorithm is used to find the optimal assignment in O(n³) time,
        with n being the maximum of periods and total capacity units
        plus idle nodes.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        # Build the list of product copies (right side nodes)
        copy_start = [0] * N  # starting index of copies for each product
        idx = 0
        for i in range(N):
            copy_start[i] = idx
            idx += capacities[i]
        total_copies = idx

        # Add dummy copies for idle periods
        num_dummies = T
        dummy_start = total_copies
        total_columns = total_copies + num_dummies

        # Determine size of Hungarian matrix
        n = max(T, total_columns)

        # Build cost matrix (to minimize cost = -profit)
        # Initialize with zeros
        cost = [[0.0] * n for _ in range(n)]

        # Fill actual costs
        for t in range(T):
            for i in range(N):
                if capacities[i] == 0:
                    continue
                profit = prices[i] * probs[t][i]
                for copy_idx in range(capacities[i]):
                    col = copy_start[i] + copy_idx
                    cost[t][col] = -profit  # negative for maximization

        # Dummy columns already zero cost (idle yields 0 revenue)

        # Hungarian algorithm (Kuhn-Munkres) for rectangular matrix
        # Implementation from Wikipedia, adapted for cost matrix
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            j0 = 0
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
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

        # Build assignment result
        assignment = [-1] * T  # default idle
        for j in range(1, n + 1):
            i = p[j]
            if i == 0 or i > T:
                continue
            t = i - 1
            col = j - 1
            if col < total_copies:
                # map copy to product
                # find i such that copy_start[i] <= col < copy_start[i]+capacities[i]
                # binary search over copy_start
                lo, hi = 0, N - 1
                prod = None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    start = copy_start[mid]
                    end = start + capacities[mid]
                    if col < start:
                        hi = mid - 1
                    elif col >= end:
                        lo = mid + 1
                    else:
                        prod = mid
                        break
                if prod is not None:
                    assignment[t] = prod
            else:
                assignment[t] = -1  # idle

        return assignment
