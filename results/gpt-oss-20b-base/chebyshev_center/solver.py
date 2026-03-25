from typing import Any, Dict, List
import numpy as np
import scipy.optimize as opt


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Chebyshev center problem via a linear program.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'a' and 'b' defining the polyhedron
            constraints a_i^T x <= b_i.

        Returns
        -------
        dict
            Dictionary with key 'solution' containing the optimal center x.
        """
        # Convert to NumPy arrays
        a = np.asarray(problem["a"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # Dimensions
        m, n = a.shape

        # Norms of constraint rows (||a_i||_2)
        a_norm = np.linalg.norm(a, axis=1)

        # Objective: maximize r, i.e. minimize -r
        c = np.zeros(n + 1, dtype=float)
        c[-1] = -1.0

        # Inequality constraints A_ub * [x; r] <= b_ub
        # a_i^T x + ||a_i|| r <= b_i
        A_ub = np.hstack([a, a_norm.reshape(-1, 1)])
        b_ub = b.copy()

        # Add non-negativity constraint on r: -r <= 0
        A_ub = np.vstack([A_ub, np.zeros((1, n + 1))])
        A_ub[-1, -1] = -1.0
        b_ub = np.concatenate([b_ub, [0.0]])

        # Solve LP using the HiGHS solver (fast and robust)
        res = opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            bounds=(None, None),  # x free, r free initially
            method="highs",
        )

        if not res.success:
            raise RuntimeError(f"Linprog did not converge: {res.message}")

        # Extract center x (first n components)
        x_opt = res.x[:n]

        return {"solution": x_opt.tolist()}
