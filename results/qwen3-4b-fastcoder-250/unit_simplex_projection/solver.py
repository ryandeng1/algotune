import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        y = np.array(problem["y"])
        n = len(y)
        sorted_y = np.sort(y)[::-1]
        cumsum_y = np.cumsum(sorted_y) - 1
        
        low, high = 0, n - 1
        while low <= high:
            mid = (low + high) // 2
            if (mid + 1) * sorted_y[mid] > cumsum_y[mid]:
                low = mid + 1
            else:
                high = mid - 1
        rho = high
        theta = cumsum_y[rho] / (rho + 1)
        x = np.maximum(y - theta, 0)
        return {"solution": x.tolist()}
