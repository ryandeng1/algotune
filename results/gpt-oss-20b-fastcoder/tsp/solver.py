from functools import lru_cache
from typing import List

class Solver:
    """
    Implements an optimal TSP solver using the Held‑Karp dynamic programming algorithm.
    The algorithm is O(n^2 2^n) which is fast enough for n <= 20 and is far faster
    than a generic CP-SAT formulation for the same problem size.
    """
    def solve(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        # Pre‑compute a list of neighbours for each node to speed up iteration
        neighbours = [list(range(n)) for _ in range(n)]

        @lru_cache(maxsize=None)
        def dp(mask: int, i: int) -> int:
            """
            Returns the minimal cost of visiting all cities in `mask` and ending at city `i`.
            `mask` is a bitmask where bit `k` is 1 iff city `k` has been visited.
            """
            if mask == (1 << i):
                # only city i is visited -> cost is 0 (starting at city i)
                return 0
            # Remove city i from mask to form submask of remaining cities
            submask = mask ^ (1 << i)
            # Iterate over all possible previous cities j in submask
            min_cost = float("inf")
            for j in neighbours[i]:
                if submask & (1 << j):
                    cost = dp(submask, j) + problem[j][i]
                    if cost < min_cost:
                        min_cost = cost
            return min_cost

        @lru_cache(maxsize=None)
        def path_backtrack(mask: int, i: int) -> List[int]:
            """
            Reconstructs the optimal path for state (mask, i).
            """
            if mask == (1 << i):
                return [i]
            submask = mask ^ (1 << i)
            best_prev = None
            best_cost = float("inf")
            for j in neighbours[i]:
                if submask & (1 << j):
                    cost = dp(submask, j) + problem[j][i]
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = j
            return path_backtrack(submask, best_prev) + [i]

        # Full mask includes all cities
        full_mask = (1 << n) - 1
        # Compute optimal tour starting at 0, ending at 0
        best_tour = None
        best_cost = float("inf")
        for i in range(1, n):
            cost = dp(full_mask, i) + problem[i][0]
            if cost < best_cost:
                best_cost = cost
                best_tour = path_backtrack(full_mask, i)

        if best_tour is None:
            return []

        # Insert start city 0 at beginning and end
        return [0] + best_tour + [0]