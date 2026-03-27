import cvxpy as cp
import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a robust linear program using CVXPY.

        Parameters
        ----------
        problem: dict[str, np.ndarray]
            Dictionary containing the problem data:
            - ``c``:     objective coefficients (1‑D array)
            - ``b``:     right‑hand side scalars (1‑D array)
            - ``P``:     list of m SPD matrices (each 2‑D array)
            - ``q``:     list of m vectors (each 1‑D array)

        Returns
        -------
        dict[str, Any]
            Dictionary with the optimal objective value and solution vector ``x``.
            If the problem is infeasible or an error occurs, ``objective_value`` is
            ``float('inf')`` and ``x`` is ``NaN``.
        """
        # Convert inputs to contiguous numpy arrays for fast indexing
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)

        n = c.size          # #decision variables
        m = b.size          # #SOC constraints

        x = cp.Variable(n)

        # Build constraints efficiently
        cons = [cp.SOC(b[i] - q[i] @ x, P[i] @ x) for i in range(m)]

        prob = cp.Problem(cp.Minimize(c @ x), cons)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {"objective_value": float("inf"),
                        "x": np.full(n, np.nan)}
            return {"objective_value": float(prob.value),
                    "x": np.asarray(x.value, dtype=np.float64)}
        except Exception:
            # Any exception from cvxpy is treated as an infeasible/unsolvable case
            return {"objective_value": float("inf"),
                    "x": np.full(n, np.nan)}