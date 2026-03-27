from typing import Any
import heapq

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Computes the 1‑dimensional Wasserstein distance (Earth Mover's Distance)
        between two discrete probability distributions `u` and `v` defined on
        the same lattice of positions 1..n.

        The implementation runs in O(n) time and O(1) additional space, using
        the fact that the optimal transport plan for a 1‑D histogram is the
        cumulative difference between the two histograms.

        :param problem: a dict with keys `u` and `v` containing weight lists.
        :return: the Wasserstein distance as a float.
        """
        u = problem["u"]
        v = problem["v"]
        if len(u) != len(v):
            raise ValueError("Distributions must have the same length.")
        n = len(u)

        # Accumulate the difference between the cumulative distributions
        diff = 0.0
        total = 0.0
        for i in range(n):
            diff += u[i] - v[i]
            total += abs(diff)
        return total