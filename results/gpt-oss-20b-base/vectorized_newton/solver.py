import numpy as np
from scipy.optimize import newton
from numba import njit

# --------------------------------------------------------------------------- #
#  Accelerated polynomial evaluation (numba JIT)
# --------------------------------------------------------------------------- #
@njit
def _task_f_vec(x, a0, a1, a2, a3, a4, a5):
    return (a1 * x * (a0 + a5 * x) - a4) * (a0 + a2 * x) + a3

@njit
def _task_f_vec_prime(x, a0, a1, a2, a3, a4, a5):
    return (a1 * (a0 + 2.0 * a5 * x) - a2 * a4) * (a0 + a2 * x) \
           + (a1 * x * (a0 + a5 * x) - a4) * a2

# --------------------------------------------------------------------------- #
#  Solver class
# --------------------------------------------------------------------------- #
class Solver:
    def __init__(self):
        self.a2 = 1e-9
        self.a3 = 0.004
        self.a4 = 10.0
        self.a5 = 0.27456

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        # Fast extraction of numpy arrays
        x0_arr = np.asarray(problem['x0'], dtype=np.float64)
        a0_arr = np.asarray(problem['a0'], dtype=np.float64)
        a1_arr = np.asarray(problem['a1'], dtype=np.float64)

        n = x0_arr.shape[0]
        if a0_arr.shape[0] != n or a1_arr.shape[0] != n:
            return {'roots': []}

        # Local references to avoid global lookups
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5

        # Wrapper functions that capture the constant parameters
        def f(x, a0, a1):
            return _task_f_vec(x, a0, a1, a2, a3, a4, a5)

        def fp(x, a0, a1):
            return _task_f_vec_prime(x, a0, a1, a2, a3, a4, a5)

        # Apply vectorised Newton
        roots_arr = newton(f, x0_arr, fprime=fp, args=(a0_arr, a1_arr))

        # Ensure we have a 1‑D array of length n
        if roots_arr.shape != (n,):
            # Fallback for single scalar result
            roots_arr = np.full(n, np.nan, dtype=np.float64)

        # Convert to list for the required return format
        return {'roots': roots_arr.tolist()}