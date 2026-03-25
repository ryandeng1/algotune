import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = alpha.size

        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        sorted_idx = np.argsort(alpha)
        a_sorted = alpha[sorted_idx]
        prefix = np.cumsum(a_sorted)
        
        w = None
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break
        if w is None:
            w = (P_total + prefix[-1]) / n
        
        x_opt = np.maximum(0.0, w - alpha)
        total_x = np.sum(x_opt)
        if total_x > 1e-9:
            scaling = P_total / total_x
            x_opt = x_opt * scaling
        else:
            scaling = 1.0
        
        safe_x = np.maximum(x_opt, 0.0)
        capacity = np.sum(np.log(alpha + safe_x))
        
        return {
            "x": x_opt.tolist(),
            "Capacity": float(capacity)
        }
