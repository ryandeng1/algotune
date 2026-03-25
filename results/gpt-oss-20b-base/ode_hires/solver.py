# solver.py
from __future__ import annotations
from typing import Any, List, Dict
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """
    Efficient stiff solver for the HIRES kinetic system.
    Uses SciPy's solve_ivp with the 'BDF' method and tight tolerances.
    """

    @staticmethod
    def _make_func(constants: List[float]):
        """
        Return a vector field function that captures the constants.
        """
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants

        def f(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            y6y8 = y6 * y8  # reuse product
            return np.array(
                [
                    -c1 * y1 + c2 * y2 + c3 * y3 + c4,
                    c1 * y1 - c5 * y2,
                    -c6 * y3 + c2 * y4 + c7 * y5,
                    c3 * y2 + c1 * y3 - c8 * y4,
                    -c9 * y5 + c2 * y6 + c2 * y7,
                    -c10 * y6y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7,
                    c10 * y6y8 - c12 * y7,
                    -c10 * y6y8 + c12 * y7,
                ],
                dtype=float,
            )

        return f

    @staticmethod
    def _solve_problem(problem: Dict[str, Any]) -> np.ndarray:
        """
        Internal solver that returns the final state vector.
        """
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        constants = np.asarray(problem["constants"], dtype=float)

        f = Solver._make_func(constants)

        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="BDF",
            rtol=1e-10,
            atol=1e-9,
            vectorized=False,
            max_step=np.inf,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.y[:, -1]

    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Public entry point. Returns the solution at the final time as a list.
        """
        y_final = Solver._solve_problem(problem)
        return y_final.tolist()
