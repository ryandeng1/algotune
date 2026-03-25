import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem["v"])
        k = problem["k"]
        u = np.abs(v)
        mu = np.sort(u)[::-1]
        cumsum = np.cumsum(mu)
        t_j = (cumsum - k) / (np.arange(len(mu)) + 1)
        mask = mu < t_j
        j = np.where(mask)[0][0]
        theta = t_j[j]
        solution = np.where(u > theta, u - theta, 0) * np.sign(v)
        return {"solution": solution.tolist()}
