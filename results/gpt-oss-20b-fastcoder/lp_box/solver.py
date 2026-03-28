from typing import Any, List, Dict
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP box problem using SciPy's linprog for maximum speed.
        The problem is to minimize c^T x subject to:
            A x <= b
            0 <= x <= 1
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # `linprog` expects the objective in the form: minimize c^T x
        # with bounds per variable. Here all variables have bounds (0,1).
        bounds = [(0.0, 1.0)] * c.shape[0]

        # SciPy's linprog uses the "highs" solver by default (fast and robust).
        res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if res.success:
            # Ensure the solution is real and finite
            x = np.asarray(res.x, dtype=np.float64)
            # Clip tiny negative values due to numerical errors
            x = np.clip(x, 0.0, 1.0)
            return {"solution": x.tolist()}
        else:
            raise RuntimeError(f"LP optimization failed: {res.message}")