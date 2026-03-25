import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        y = np.array(problem["y"])
        y = y.flatten()
        n = len(y)
        sorted_y = np.sort(y)[::-1]
        cumsum_y = np.cumsum(sorted_y) - 1
        idx = np.arange(1, n + 1)
        left = idx * sorted_y
        mask = left > cumsum_y
        rho = np.where(mask)[0][-1] if np.any(mask) else 0
        theta = cumsum_y[rho] / (rho + 1)
        x = np.maximum(y - theta, 0)
        return {"solution": x.tolist()}
