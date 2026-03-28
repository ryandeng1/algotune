from typing import List, Dict
import numpy as np
from scipy.optimize import newton

# ----------------------------------------------------------------------
# The analytic expressions used by the Newton solver
# ----------------------------------------------------------------------
def _task_f_vec(x: np.ndarray, a0: np.ndarray, a1: np.ndarray,
                a2: float, a3: float, a4: float, a5: float) -> np.ndarray:
    """Vectorised evaluation of the scalar function for each element."""
    # The function is f(x) = a0 * sin(x) + a1 * x**2 + a2 * x**4 + a3 * exp(x) + a4 * log(1 + x**2) + a5
    return (a0 * np.sin(x) +
            a1 * x**2 +
            a2 * x**4 +
            a3 * np.exp(x) +
            a4 * np.log1p(x**2) +
            a5)

def _task_f_vec_prime(x: np.ndarray, a0: np.ndarray, a1: np.ndarray,
                      a2: float, a3: float, a4: float, a5: float) -> np.ndarray:
    """Vectorised evaluation of the derivative."""
    return (a0 * np.cos(x) +
            2 * a1 * x +
            4 * a2 * x**3 +
            a3 * np.exp(x) +
            2 * a4 * x / (1 + x**2))

# ----------------------------------------------------------------------
# Solver implementation
# ----------------------------------------------------------------------
class Solver:
    # constant coefficients
    a2: float = 1e-9
    a3: float = 0.004
    a4: float = 10.0
    a5: float = 0.27456

    # Bound the initial guesses to avoid overflow in exp/log
    _inv_max_guess: float = 1.0 / 1000.0  # safe range for exp and log

    def solve(self, problem: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Find roots of the vectorised non‑linear system with Newton’s method.

        Parameters
        ----------
        problem
            Dictionary with keys ``x0``, ``a0`` and ``a1``.
            All must contain the same length ``n``.  The solution
            contains ``n`` roots.  ``NaN`` is returned for elements
            where the solver failed.

        Returns
        -------
        result
            Dictionary with the key ``roots`` holding a list of
            floating point numbers.
        """
        try:
            x0 = np.asarray(problem["x0"], dtype=np.float64)
            a0 = np.asarray(problem["a0"], dtype=np.float64)
            a1 = np.asarray(problem["a1"], dtype=np.float64)
        except Exception:
            return {"roots": []}

        if not (x0.shape == a0.shape == a1.shape):
            return {"roots": []}

        # Protect against overflow in exp and log by clipping
        # Only needed for very large initial guesses
        x0 = np.clip(x0, -1 / self._inv_max_guess, 1 / self._inv_max_guess)

        args = (a0, a1, self.a2, self.a3, self.a4, self.a5)

        # ``newton`` accepts vectorised functions and works on the whole array
        try:
            roots = newton(self._func,
                           x0,
                           fprime=self._fprime,
                           args=args,
                           maxiter=200,
                           tol=1e-12,
                           rtol=0)
        except Exception:
            # In the unlikely event that Newton fails,
            # return NaN for all entries
            return {"roots": [float("nan")] * len(x0)}

        # ``roots`` may be a scalar if only one element was given
        if np.isscalar(roots):
            roots = np.array([roots], dtype=np.float64)

        # Convert to regular Python list (maintains shape)
        return {"roots": roots.tolist()}

    # ----------------------------------------------------------------------
    # Helper wrappers around the function and its derivative
    # ----------------------------------------------------------------------
    @staticmethod
    def _func(x: np.ndarray,
              a0: np.ndarray,
              a1: np.ndarray,
              a2: float,
              a3: float,
              a4: float,
              a5: float) -> np.ndarray:
        return _task_f_vec(x, a0, a1, a2, a3, a4, a5)

    @staticmethod
    def _fprime(x: np.ndarray,
                a0: np.ndarray,
                a1: np.ndarray,
                a2: float,
                a3: float,
                a4: float,
                a5: float) -> np.ndarray:
        return _task_f_vec_prime(x, a0, a1, a2, a3, a4, a5)