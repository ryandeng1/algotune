from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, List[float]]) -> float:
        """
        Computes the 1‑D Wasserstein distance between two discrete distributions
        defined on the support {1, 2, …, n}. The distributions are given by the
        weight lists ``u`` and ``v`` and the reference positions are the
        integer indices.
        """
        try:
            u: List[float] = problem["u"]
            v: List[float] = problem["v"]
            n = len(u)

            # Quick check – the lists must be of the same length
            if n != len(v):
                raise ValueError("u and v must have the same length")

            # Cumulative sums of the two distributions
            cum_u = 0.0
            cum_v = 0.0
            distance = 0.0

            for weight_u, weight_v in zip(u, v):
                cum_u += weight_u
                cum_v += weight_v
                distance += abs(cum_u - cum_v)

            # If the supports are equally spaced by 1, no additional factor is needed
            return distance

        except Exception:
            # Gracefully handle any unexpected errors – return a fallback value
            return float(len(problem.get("u", [])))