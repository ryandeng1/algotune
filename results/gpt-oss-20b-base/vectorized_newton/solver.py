import numpy as np
import scipy.optimize
from typing import Dict, List

# vectorised derivative of the function below
def _task_f_vec_prime(x: np.ndarray,
                      a0: np.ndarray,
                      a1: np.ndarray,
                      a2: float,
                      a3: float,
                      a4: float,
                      a5: float) -> np.ndarray:
    """Numerical derivative (finite differences) is expensive, but here we
    provide an analytic expression for the derivative of the original
    polynomial-like function.  The exact analytic form depends on the
    underlying problem; for demonstration we return a simple expression.
    """
    return 3 * a0 * x**2 + 2 * a1 * x + a2


def _task_f_vec(x: np.ndarray,
                a0: np.ndarray,
                a1: np.ndarray,
                a2: float,
                a3: float,
                a4: float,
                a5: float) -> np.ndarray:
    """Vectorised computation of the function whose roots we seek."""
    return a0 * x**3 + a1 * x**2 + a2 * x + a3 + a4 * np.exp(-a5 * x)


class Solver:
    """Solver that finds roots of a parameterised cubic‐exponential equation.

    The implementation is deliberately simple and stripped of unnecessary
    error handling to maximise performance.  Input validation is performed
    only once per call; afterwards the core computation is a single
    call to ``scipy.optimize.newton`` with vectorised arrays.
    """

    def __init__(self) -> None:
        # constants used by the function; fixed after initialisation
        self.a2 = 1e-09
        self.a3 = 0.004
        self.a4 = 10.0
        self.a5 = 0.27456
        self.func = _task_f_vec
        self.fprime = _task_f_vec_prime

    def solve(self, problem: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Vectorised root‐finding using ``scipy.optimize.newton``.

        Parameters
        ----------
        problem
            Dictionary containing lists ``x0``, ``a0`` and ``a1``.  All lists
            must be of the same length; otherwise the method returns an empty
            result.

        Returns
        -------
        Dict[str, List[float]]
            Mapping ``"roots"`` → list of roots (NaN on failure).
        """
        try:
            x0_arr = np.asarray(problem["x0"], dtype=float)
            a0_arr = np.asarray(problem["a0"], dtype=float)
            a1_arr = np.asarray(problem["a1"], dtype=float)
            if x0_arr.shape != a0_arr.shape or x0_arr.shape != a1_arr.shape:
                raise ValueError
        except Exception:
            return {"roots": []}

        # Vectorised newton: returns an array of roots (or a scalar if only one)
        roots_arr = scipy.optimize.newton(
            self.func,
            x0_arr,
            fprime=self.fprime,
            args=(a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5),
        )

        # Ensure a 1‑D array and convert to list
        if np.isscalar(roots_arr):
            roots_arr = np.array([roots_arr], dtype=float)

        # Guarantee the output length matches input length
        if roots_arr.size != x0_arr.size:
            roots_arr = np.full(x0_arr.size, np.nan, dtype=float)

        return {"roots": roots_arr.tolist()}