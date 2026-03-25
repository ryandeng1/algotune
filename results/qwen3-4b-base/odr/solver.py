import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        x = np.asarray(problem["x"])
        y = np.asarray(problem["y"])
        sx = np.asarray(problem["sx"])
        sy = np.asarray(problem["sy"])
        
        w = 1.0 / (sx**2 + sy**2)
        
        sum_w = np.sum(w)
        sum_w_x = np.sum(w * x)
        sum_w_y = np.sum(w * y)
        sum_w_x2 = np.sum(w * x**2)
        sum_w_x_y = np.sum(w * x * y)
        
        numerator = sum_w_x_y - (sum_w_x * sum_w_y) / sum_w
        denominator = sum_w_x2 - (sum_w_x**2) / sum_w
        a = numerator / denominator
        
        b = (sum_w_y - a * sum_w_x) / sum_w
        
        return {"beta": [a, b]}
