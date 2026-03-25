from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[float]]) -> Any:
        """
        Compute the 1‑dimensional Wasserstein distance between two discrete
        probability distributions defined on the integers 1..n.

        For a 1‑D space with equal spacing the optimal coupled plan is obtained
        by sorting the mass and the distance reduces to the L¹ distance
        between the cumulative distribution functions.

        This implementation uses cumulative sums and runs in O(n) time
        without external dependencies beyond the Python standard library.
        """
        u = problem.get("u", [])
        v = problem.get("v", [])

        # Basic validation: lengths must match
        if len(u) != len(v):
            # Fall back to a safe default (worst case distance)
            return float(len(u))

        n = len(u)

        # Convert to lists of floats for consistency
        u = [float(x) for x in u]
        v = [float(x) for x in v]

        # Compute cumulative sums
        cum_u = 0.0
        cum_v = 0.0
        dist = 0.0

        for ui, vi in zip(u, v):
            cum_u += ui
            cum_v += vi
            dist += abs(cum_u - cum_v)

        return float(dist)
