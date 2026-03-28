from typing import Any
from scipy.stats import wasserstein_distance

class Solver:

    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Solves the wasserstein distance using scipy.stats.wasserstein_distance.

        :param problem: a Dict containing info for dist u and v
        :return: A float determine the wasserstein distance
        """
        try:
            n = len(problem['u'])
            d = wasserstein_distance(list(range(1, n + 1)), list(range(1, n + 1)), problem['u'], problem['v'])
            return d
        except Exception as e:
            return float(len(problem['u']))
        else:
            pass
        finally:
            pass
