import numpy as np

class Solver:
    def __init__(self, a2, a3, a4, a5):
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def solve(self, problem):
        try:
            x0 = np.array(problem["x0"])
            a0 = np.array(problem["a0"])
            a1 = np.array(problem["a1"])
            n = len(x0)
            if len(a0) != n or len(a1) != n:
                return {"roots": [np.nan] * n}
        except Exception:
            return {"roots": [np.nan] * n}
        
        x = x0.copy()
        max_iter = 3
        tol = 1e-8
        for _ in range(max_iter):
            exp_term = np.exp((a0 + x * self.a3) / self.a5)
            term1 = self.a2 * (exp_term - 1)
            term2 = (a0 + x * self.a3) / self.a4
            f_val = a1 - term1 - term2 - x
            
            f_prime_val = -self.a2 * (self.a3 / self.a5) * exp_term - (self.a3 / self.a4 + 1)
            
            valid_mask = np.isfinite(f_prime_val) & (f_prime_val != 0)
            step = np.zeros_like(f_val)
            step[valid_mask] = f_val[valid_mask] / f_prime_val[valid_mask]
            
            x = x - step
            
            if np.all(np.abs(step[valid_mask]) < tol):
                break
        
        return {"roots": x.tolist()}
