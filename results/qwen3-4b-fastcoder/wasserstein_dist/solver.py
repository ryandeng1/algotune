from typing import Any
from scipy.stats import wasserstein_distance

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        try:
            n = len(problem["u"])
            indices = list(range(1, n + 1))
            d = wasserstein_distance(indices, indices, problem["u"], problem["v"])
            return d
        except Exception as e:
            return float(len(problem["u"]))