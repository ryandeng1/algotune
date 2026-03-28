from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
import numba

# JIT‑compiled RHS for speed
@numba.njit
def _f_numba(y: np.ndarray, constants: np.ndarray) -> np.ndarray:
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
    f2 = c1 * y1 - c5 * y2
    f3 = -c6 * y3 + c2 * y4 + c7 * y5
    f4 = c3 * y2 + c1 * y3 - c8 * y4
    f5 = -c9 * y5 + c2 * y6 + c2 * y7
    f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
    f7 = c10 * y6 * y8 - c12 * y7
    f8 = -c10 * y6 * y8 + c12 * y7
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        constants = np.asarray(problem["constants"], dtype=np.float64)

        # Wrapper to satisfy the signature required by solve_ivp
        def rhs(t, y):
            return _f_numba(y, constants)

        # Use a stiff solver with high precision but no dense output
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-10,
            atol=1e-9,
            t_eval=None,
            dense_output=False,
        )
        return sol