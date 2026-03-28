from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP box problem using SciPy's linprog which is
        considerably faster than CVXPY for small- to medium-sized
        problems.

        :param problem: A dictionary containing:
            - c: objective coefficients
            - A: inequality matrix
            - b: inequality bounds
        :return: A dictionary with the optimal solution vector.
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # Bounds: 0 <= x_i <= 1 for all i
        bounds = [(0.0, 1.0)] * c.size

        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if res.status != 0:
            raise RuntimeError(f"LP did not converge: {res.message}")

        return {"solution": res.x.tolist()}