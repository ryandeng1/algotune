from typing import List, Dict

class Solver:
    def solve(self, problem: Dict[str, List[float]]) -> float:
        """
        Compute the 1‑dimensional Wasserstein distance
        for two distributions defined on the discrete grid 1…n.
        The distances are given by their probability mass functions in
        problem['u'] and problem['v'].
        """
        u = problem.get("u")
        v = problem.get("v")
        if not u or not v or len(u) != len(v):
            return float(len(u) if u else 0)

        # cumulative sums
        cum_u = 0.0
        cum_v = 0.0
        dist = 0.0
        for pu, pv in zip(u, v):
            cum_u += pu
            cum_v += pv
            dist += abs(cum_u - cum_v)
        return dist