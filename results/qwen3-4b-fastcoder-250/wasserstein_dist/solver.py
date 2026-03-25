import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        u = problem["u"]
        v = problem["v"]
        cum_u = np.cumsum(u)
        cum_v = np.cumsum(v)
        return np.sum(np.abs(cum_u - cum_v))
