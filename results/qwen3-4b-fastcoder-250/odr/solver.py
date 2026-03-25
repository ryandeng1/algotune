import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        x = np.array(problem["x"])
        y = np.array(problem["y"])
        sx = np.array(problem["sx"])
        sy = np.array(problem["sy"])
        
        weights = 1.0 / (sx**2 + sy**2)
        sum_w = np.sum(weights)
        sum_w_x = np.sum(weights * x)
        sum_w_y = np.sum(weights * y)
        sum_w_x2 = np.sum(weights * x**2)
        sum_w_xy = np.sum(weights * x * y)
        
        a = sum_w_xy / sum_w_x2
        b = (sum_w_y / sum_w) - a * (sum_w_x / sum_w)
        
        return {"beta": [a, b]}
