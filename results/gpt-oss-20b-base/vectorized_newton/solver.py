import numpy as np

# Fixed parameters for the task
A2 = 1.0   # example value, will be overwritten by problem dict if present
A3 = 1.0
A4 = 1.0
A5 = 1.0

def func(x, a0, a1, a2, a3, a4, a5):
    """Compute f(x) = a1 - a2*(exp((a0+x*a3)/a5)-1) - (a0+x*a3)/a4 - x"""
    return a1 - a2 * (np.exp((a0 + x * a3) / a5) - 1) - (a0 + x * a3) / a4 - x

def fprime(x, a0, a1, a2, a3, a4, a5):
    """Derivative of f with respect to x"""
    exp_term = np.exp((a0 + x * a3) / a5)
    return -a2 * a3 / a5 * exp_term - a3 / a4 - 1.0

class Solver:
    def __init__(self, *, a2=1.0, a3=1.0, a4=1.0, a5=1.0):
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        # Convert inputs to numpy arrays
        try:
            x = np.asarray(problem["x0"], dtype=float)
            a0 = np.asarray(problem["a0"], dtype=float)
            a1 = np.asarray(problem["a1"], dtype=float)
        except Exception:
            return {"roots": []}

        n = x.size
        # Ensure all arrays have the same length
        if a0.size != n or a1.size != n:
            return {"roots": []}

        # Newton-Raphson parameters
        max_iter = 20
        tol = 1e-12

        # Initialize roots with initial guesses
        roots = x.copy()

        for _ in range(max_iter):
            f_val = func(roots, a0, a1, self.a2, self.a3, self.a4, self.a5)
            if np.all(np.abs(f_val) < tol):
                break
            fp_val = fprime(roots, a0, a1, self.a2, self.a3, self.a4, self.a5)
            # Avoid division by zero
            mask = fp_val != 0
            if not np.all(mask):
                # Replace zero derivatives with small epsilon to continue
                fp_val[~mask] = 1e-14
            step = f_val / fp_val
            roots[mask] -= step[mask]
            # For any remaining NaNs or infs, set to NaN
            roots[~np.isfinite(roots)] = np.nan

        return {"roots": roots.tolist()}
