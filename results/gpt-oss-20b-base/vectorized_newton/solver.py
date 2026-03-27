import numpy as np
from typing import Dict, List


class Solver:
    """
    Finds roots for a set of polynomials of the form
        f(x) = a0 + a1*x + ... + a5*x**5
    using an explicit Newton iteration.
    """

    # Polynomial coefficients up to 5th degree are passed when creating the instance.
    def __init__(self, a2: float, a3: float, a4: float, a5: float) -> None:
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    # ----------------------------------------------------------------------
    # Public API ------------------------------------------------------------
    # ----------------------------------------------------------------------
    def solve(self, problem: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        :param problem: Dictionary with keys 'x0', 'a0', 'a1' mapping to lists of
                        the same length.
        :return: Dictionary with key 'roots', containing a list of roots
                        (float).  On failure a NaN is returned for that entry.
        """
        try:
            x0 = np.asarray(problem["x0"], dtype=float)
            a0 = np.asarray(problem["a0"], dtype=float)
            a1 = np.asarray(problem["a1"], dtype=float)

            if x0.shape != a0.shape or x0.shape != a1.shape:
                raise ValueError("Input lists have mismatched shapes")

        except Exception:
            # If we cannot obtain arrays we return an empty list
            return {"roots": []}

        # Perform Newton iterations for each element independently
        roots = self._newton_iterate(x0, a0, a1)

        return {"roots": roots.tolist()}

    # ----------------------------------------------------------------------
    # Internal helpers ------------------------------------------------------
    # ----------------------------------------------------------------------
    def _newton_iterate(
        self,
        x0: np.ndarray,
        a0: np.ndarray,
        a1: np.ndarray,
        *,
        tol: float = 1e-12,
        max_iter: int = 100,
    ) -> np.ndarray:
        """
        Implements Newton's method for each element of x0.
        """
        n = x0.size
        roots = np.empty(n, dtype=float)
        roots[:] = np.nan  # default NaN for failures

        for i in range(n):
            xi = x0[i]
            try:
                for _ in range(max_iter):
                    # f(x) = a0 + a1*x + a2*x**2 + ... + a5*x**5
                    f_val = (
                        a0[i]
                        + a1[i] * xi
                        + self.a2 * xi * xi
                        + self.a3 * xi ** 3
                        + self.a4 * xi ** 4
                        + self.a5 * xi ** 5
                    )
                    # f'(x) = a1 + 2*a2*x + 3*a3*x**2 + 4*a4*x**3 + 5*a5*x**4
                    fp_val = (
                        a1[i]
                        + 2 * self.a2 * xi
                        + 3 * self.a3 * xi * xi
                        + 4 * self.a4 * xi ** 3
                        + 5 * self.a5 * xi ** 4
                    )
                    if fp_val == 0:
                        break  # derivative zero, cannot proceed

                    xi_next = xi - f_val / fp_val
                    if abs(xi_next - xi) < tol:
                        roots[i] = xi_next
                        break
                    xi = xi_next

                # If we exit the loop without meeting tolerance, keep NaN
            except Exception:
                # Any unexpected problem results in NaN for this entry
                pass

        return roots