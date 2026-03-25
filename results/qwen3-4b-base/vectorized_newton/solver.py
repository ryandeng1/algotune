import numpy as np

class Solver:
    def __init__(self, a2, a3, a4, a5):
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        try:
            x0_arr = np.array(problem["x0"])
            a0_arr = np.array(problem["a0"])
            a1_arr = np.array(problem["a1"])
            n = len(x0_arr)
            if len(a0_arr) != n or len(a1_arr) != n:
                return {"roots": [float("nan")] * n}
        except Exception as e:
            return {"roots": [float("nan")] * n}

        x0_arr = x0_arr.astype(float)
        a0_arr = a0_arr.astype(float)
        a1_arr = a1_arr.astype(float)
        x = x0_arr.copy()
        tol = 1e-8
        max_iter = 100
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5

        a3_div_a5 = a3 / a5
        a3_div_a4 = a3 / a4

        for it in range(max_iter):
            term = (a0_arr + x * a3) / a5
            exp_term = np.exp(term)
            f_val = a1_arr - a2 * (exp_term - 1) - (a0_arr + x * a3) / a4 - x
            f_prime_val = -a2 * exp_term * a3_div_a5 - a3_div_a4 - 1

            f_prime_mask = np.abs(f_prime_val) < 1e-12
            if np.any(f_prime_mask):
                x[f_prime_mask] = np.nan
                continue

            x_new = x - f_val / f_prime_val
            converged = np.abs(f_val) < tol
            if np.all(converged):
                x = x_new
                break

            x = x_new

        return {"roots": x.tolist()}
