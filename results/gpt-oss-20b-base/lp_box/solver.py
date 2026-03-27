from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP box problem using SciPy's `linprog` for speed.

        :param problem: A dictionary containing:
                        - "c": cost vector
                        - "A": constraint matrix
                        - "b": right‑hand side vector
        :return: Dictionary with key "solution" holding the optimal variable values.
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # Bounds: 0 <= x_i <= 1
        bounds = [(0.0, 1.0)] * c.size

        # linprog expects a mi p x n matrix for <= constraints
        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if not res.success:
            raise ValueError("Linear program did not converge to optimal solution.")

        return {"solution": res.x.tolist()}