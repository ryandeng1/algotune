from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        y = np.array(problem.get("y"))
        y = y.flatten()
        n = len(y)
        
        if n == 0:
            return {"solution": []}
        
        sorted_y = np.sort(y)[::-1]
        cumsum_y = np.cumsum(sorted_y) - 1
        
        low, high = 0, n - 1
        rho = 0
        while low <= high:
            mid = (low + high) // 2
            threshold = cumsum_y[mid] / (mid + 1)
            if sorted_y[mid] > threshold:
                rho = mid
                low = mid + 1
            else:
                high = mid - 1
        
        theta = cumsum_y[rho] / (rho + 1)
        x = np.maximum(y - theta, 0)
        return {"solution": x.tolist()}