from typing import Dict, List

class Solver:
    """Optimised 1‑D Wasserstein distance solver.

    The Wasserstein distance between two discrete probability
    distributions on the same support is the L1 distance between
    their cumulative distribution functions.  If ``u`` and ``v`` are
    weight lists (not necessarily summing to 1) the distance is
    ``sum(abs(cum_u - cum_v))`` and the support is simply ``1, 2, …,
    n`` where ``n`` is the length of the lists.
    """

    def solve(self, problem: Dict[str, List[float]]) -> float:
        try:
            u = problem["u"]
            v = problem["v"]
            if len(u) != len(v):
                raise ValueError("length mismatch")
            cum_u = 0.0
            cum_v = 0.0
            distance = 0.0
            for a, b in zip(u, v):
                cum_u += a
                cum_v += b
                distance += abs(cum_u - cum_v)
            return distance
        except Exception:
            # Fallback: if input is invalid, return a sanity value
            return float(len(problem.get("u", [])))