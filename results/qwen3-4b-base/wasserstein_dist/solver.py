import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> float:
        u = problem["u"]
        v = problem["v"]
        n = len(u)
        cum_u = np.cumsum(u)
        cum_v = np.cumsum(v)
        return np.sum(np.abs(cum_u[:-1] - cum_v[:-1]))
