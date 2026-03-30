# solver.py
import numpy as np
from typing import Any, Dict, List
from scipy.optimize import linprog

class Solver:
    """
    Efficient solver for the Chebyshev center problem.
    The problem is reformulated as a standard linear program:
        maximize     r
        subject to  a_i^T x + ||a_i|| * r <= b_i   for all i
                    r >= 0
    Since all constraints and the objective are linear, SciPy's
    high‑efficiency simplex/PIVOT (Highs) solver is used.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Chebyshev center problem.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - 'a' : 2‑D array-like (m x n) data matrix
                - 'b' : 1‑D array-like (m,) RHS vector

        Returns
        -------
        dict
            Dictionary with key 'solution' containing the optimal
            x as a 1‑D list of floats.
        """
        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        m, n = a.shape
        # norms of rows of a
        norms = np.linalg.norm(a, axis=1)

        # Build linear program in standard form: min c^T z subject to A_ub z <= b_ub
        # Decision variables: z = [x_1, …, x_n, r]
        A_ub = np.column_stack((a, norms.reshape(-1, 1)))
        b_ub = b

        # Constraint r >= 0 -> -r <= 0
        A_ub = np.vstack((A_ub, np.hstack((np.zeros(n), [-1]))))
        b_ub = np.append(b_ub, 0.0)

        # Objective: maximize r -> minimize -r
        c = np.zeros(n + 1, dtype=np.float64)
        c[-1] = -1.0

        # SciPy's linprog (Highs) handles non‑negativity by default for all variables.
        # But r can be any real; we enforced r >= 0 explicitly above.
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")

        if not res.success:
            raise RuntimeError(f"Linear program did not converge: {res.message}")

        x_opt = res.x[:n]
        return {"solution": x_opt.tolist()}