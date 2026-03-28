from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        y = np.array(problem.get("y"))
        y = y.flatten()
        n = len(y)
        
        sorted_y = np.sort(y)[::-1]
        cumsum_y = np.cumsum(sorted_y) - 1
        
        indices = np.arange(n)
        condition = (indices + 1) * sorted_y > cumsum_y
        rho = np.where(condition)[0][-1]
        
        theta = cumsum_y[rho] / (rho + 1)
        x = np.maximum(y - theta, 0)
        return {"solution": x.tolist()}