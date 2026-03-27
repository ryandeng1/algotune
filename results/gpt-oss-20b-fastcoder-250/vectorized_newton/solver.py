import numpy as np
import scipy.optimize
from typing import Dict, List


class Solver:
    """
    Uses a single vectorised call to scipy.optimize.newton to find the roots of
    a quintic function:
        f(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
    The coefficients a2..a5 must be set before calling solve().
    """

    def __init__(self, a2: float, a3: float, a4: float, a5: float) -> None:
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    # ---------------------  helper to evaluate f and f'  --------------------- #
    @staticmethod
    def _func(x: np.ndarray,
              a0: np.ndarray,
              a1: np.ndarray,
              a2: float,
              a3: float,
              a4: float,
              a5: float) -> np.ndarray:
        """Vectorised evaluation of the quintic polynomial."""
        return (
            a0
            + a1 * x
            + a2 * x * x
            + a3 * x * x * x
            + a4 * x * x * x * x
            + a5 * x * x * x * x * x
        )

    @staticmethod
    def _fprime(x: np.ndarray,
                a0: np.ndarray,
                a1: np.ndarray,
                a2: float,
                a3: float,
                a4: float,
                a5: float) -> np.ndarray:
        """Vectorised evaluation of the derivative of the quintic."""
        return (
            a1
            + 2 * a2 * x
            + 3 * a3 * x * x
            + 4 * a4 * x * x * x
            + 5 * a5 * x * x * x * x
        )

    # ---------------------------  public API  --------------------------- #
    def solve(self, problem: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        :param problem: Dict containing lists of equal length:
                        - "x0": initial guesses
                        - "a0": first coefficient per element
                        - "a1": second coefficient per element
        :return: Dict with key "roots" -> list of roots (or float('nan') on failure)
        """
        try:
            x0 = np.asarray(problem["x0"], dtype=float)
            a0 = np.asarray(problem["a0"], dtype=float)
            a1 = np.asarray(problem["a1"], dtype=float)
        except Exception:
            # Invalid input
            return {"roots": []}

        n = len(x0)
        if not (len(a0) == len(a1) == n):
            return {"roots": []}

        # Prepare arguments for vectorised Newton
        f_args = (a0, a1, self.a2, self.a3, self.a4, self.a5)

        try:
            # Vectorised newton:
            roots = scipy.optimize.newton(
                self._func,
                x0,
                fprime=self._fprime,
                args=f_args,
                maxsteps=50,  # can tune if needed
                tol=1e-12,
            )

            # Ensure we get a 1‑D array of roots
            roots = np.asarray(roots, dtype=float)
            if roots.ndim == 0:  # single root when n == 1
                roots = np.full(n, roots, dtype=float)

            # If vectorised call returned a wrong length, pad with NaNs
            if len(roots) != n:
                padding = np.full(n - len(roots), np.nan, dtype=float)
                roots = np.concatenate([roots, padding])

        except Exception:
            # Anything went wrong – return NaNs for all elements
            roots = np.full(n, np.nan, dtype=float)

        return {"roots": roots.tolist()}