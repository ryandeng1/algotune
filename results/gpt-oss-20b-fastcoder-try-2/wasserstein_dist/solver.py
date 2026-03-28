from typing import Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[float]]) -> float:
        """
        Fast O(n) computation of the 1‑dimensional Wasserstein distance
        between two discrete distributions on the same equally spaced grid.
        """
        u = problem.get("u")
        v = problem.get("v")
        if u is None or v is None or len(u) != len(v):
            # fall back to a simple penalty value
            return float(len(u) if u is not None else 0)

        # cumulative sums and running total of absolute differences
        cum_u = 0.0
        cum_v = 0.0
        distance = 0.0
        for a, b in zip(u, v):
            cum_u += a
            cum_v += b
            distance += abs(cum_u - cum_v)

        return distance