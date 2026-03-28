from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the LP box problem using SciPy's linear programming routine.

        The problem is:
            minimize   cᵀ x
            subject to A x ≤ b
                       0 ≤ x ≤ 1

        :param problem: A dictionary containing keys 'c', 'A', 'b'.
        :return: Dictionary with key 'solution' containing the optimal x as a Python list.
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # SciPy's linprog expects A_ub @ x <= b_ub
        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=(0, 1),
            method="highs",
            options={"presolve": True, "time_limit": 30},
        )

        if not res.success:
            raise RuntimeError(f"LP solver failed: {res.message}")

        return {"solution": res.x.tolist()}