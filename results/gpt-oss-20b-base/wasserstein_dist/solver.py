from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Compute the 1‑D Wasserstein distance between two distributions
        `u` and `v` defined on the same support {1, …, n}.

        The distance equals the L1 distance between the cumulative
        distribution functions, which can be computed in O(n) time
        without external libraries.
        """
        u = problem.get('u', [])
        v = problem.get('v', [])
        n = len(u)
        if n == 0 or n != len(v):
            return float(n)

        cum_u = 0.0
        cum_v = 0.0
        dist = 0.0
        for i in range(n):
            cum_u += u[i]
            cum_v += v[i]
            dist += abs(cum_u - cum_v)

        return dist