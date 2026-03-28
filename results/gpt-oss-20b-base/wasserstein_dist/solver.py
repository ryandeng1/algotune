from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Computes the 1‑dimensional Wasserstein distance (Earth Mover's Distance)
        between two discrete probability distributions that are represented
        by the histograms `u` and `v`.  The positions are assumed to be
        the integer 1‑based indices of the bins.

        This implementation is O(n) and does not rely on scipy, so it is
        much faster and more deterministic than using scipy.stats.wasserstein_distance.
        """
        try:
            u = list(problem['u'])
            v = list(problem['v'])
            n = len(u)
            if len(v) != n:
                return float(n)
            cum_u = 0.0
            cum_v = 0.0
            dist = 0.0
            for i in range(n):
                cum_u += u[i]
                cum_v += v[i]
                dist += abs(cum_u - cum_v)
            return dist
        except Exception:
            return float(len(problem.get('u', [])))