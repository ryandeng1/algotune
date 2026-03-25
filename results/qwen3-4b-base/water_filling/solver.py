import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        alpha = problem["alpha"]
        P_total = problem["P_total"]
        n = len(alpha)
        
        alpha_arr = np.asarray(alpha, dtype=float)
        if n == 0 or P_total <= 0 or not np.all(alpha_arr > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        
        sorted_indices = np.argsort(alpha_arr)
        sorted_alpha = alpha_arr[sorted_indices]
        prefix = np.cumsum(sorted_alpha)
        
        w = None
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix[k-1]) / k
            if k == n or w_candidate <= sorted_alpha[k]:
                w = w_candidate
                break
        if w is None:
            w = (P_total + prefix[-1]) / n
        
        x_sorted = np.maximum(0.0, w - sorted_alpha)
        x_original = np.zeros(n)
        x_original[sorted_indices] = x_sorted
        
        total_power = np.sum(x_original)
        if total_power > 1e-9:
            scaling_factor = P_total / total_power
            x_original = x_original * scaling_factor
        
        x_original = np.maximum(x_original, 0.0)
        capacity = np.sum(np.log(alpha_arr + x_original))
        
        return {
            "x": x_original.tolist(),
            "Capacity": float(capacity)
        }
