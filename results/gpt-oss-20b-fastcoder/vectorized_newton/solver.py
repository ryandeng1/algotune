# solver.py
import numpy as np
from numba import njit, prange
from typing import Dict, List

# ------------------------------------------------------------------
# Numerical helpers – compiled with Numba to avoid Python overhead
# ------------------------------------------------------------------
@njit(parallel=True)
def _root_find(
    x0: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    a2: float,
    a3: float,
    a4: float,
    a5: float,
    max_iter: int = 50,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Parallel Newton–Raphson solver for n independent scalar equations.
    Solves the system
        f(x) = a0 + a1·x + a2·x² + a3·x³ + a4·x⁴ + a5·x⁵ = 0

    Parameters
    ----------
    x0 : nd.array
        Initial guesses.
    a0, a1, a2, a3, a4, a5
        Polynomial coefficients (a2–a5 are constants).
    max_iter : int
        Maximum number of GM iterations per root.
    tol : float
        Convergence tolerance.

    Returns
    -------
    roots : np.ndarray
        Solution array; NaN for non‑convergent cases.
    """
    n = x0.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        xi = x0[i]
        ai0 = a0[i]
        ai1 = a1[i]
        okay = False
        for _ in range(max_iter):
            # f(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
            xi2 = xi * xi
            f = ai0 + ai1 * xi + a2 * xi2 \
                + a3 * xi2 * xi + a4 * xi2 * xi2 + a5 * xi2 * xi2 * xi
            # f'(x) = a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4
            fp = ai1 + 2 * a2 * xi + 3 * a3 * xi2 + 4 * a4 * xi2 * xi + 5 * a5 * xi2 * xi2

            if fp == 0.0:
                break  # avoid division by zero

            xi_next = xi - f / fp
            if np.abs(xi_next - xi) < tol:
                xi = xi_next
                okay = True
                break
            xi = xi_next

        out[i] = xi if okay else np.nan
    return out


# ------------------------------------------------------------------
# Solver wrapper
# ------------------------------------------------------------------
class Solver:
    """
    High‑performance root finder for the polynomial

        f(x) = a0 + a1·x + a2·x² + a3·x³ + a4·x⁴ + a5·x⁵

    The solver uses Numba‑compiled code to perform a vectorised Newton–Raphson
    iteration on all provided starting points.  This avoids multiple calls to
    scipy and eliminates the heavy Python loop overhead.
    """

    def __init__(self) -> None:
        # Parameters of the polynomial – these are the same for every root
        self.a2 = 1e-9
        self.a3 = 0.004
        self.a4 = 10.0
        self.a5 = 0.27456

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, problem: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Public entry point expected by the harness.

        Parameters
        ----------
        problem : dict
            Must contain keys ``"x0"``, ``"a0"``, ``"a1"`` whose values are
            list‑like containers of equal length.

        Returns
        -------
        dict
            A mapping with a single key ``"roots"`` containing the list of
            converged roots; ``NaN`` is used for entries that failed to
            converge.
        """
        try:
            x0_arr = np.asarray(problem["x0"], dtype=np.float64)
            a0_arr = np.asarray(problem["a0"], dtype=np.float64)
            a1_arr = np.asarray(problem["a1"], dtype=np.float64)

            # Bail out early if sizes mismatch
            if x0_arr.shape != a0_arr.shape or x0_arr.shape != a1_arr.shape:
                return {"roots": []}

        except Exception:
            return {"roots": []}

        # Call the numba kernel – it returns a NumPy array
        roots_arr = _root_find(
            x0_arr,
            a0_arr,
            a1_arr,
            self.a2,
            self.a3,
            self.a4,
            self.a5,
        )

        # Convert to Python list for deterministic JSON serialisation
        roots = roots_arr.tolist()
        return {"roots": roots}