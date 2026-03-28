import numpy as np
import cvxpy as cp
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solves a robust LP with SOC constraints using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - c : ndarray, objective coefficients (n,)
            - b : ndarray, RHS scalars (m,)
            - P : ndarray, list of symmetric PSD matrices (m, n, n)
            - q : ndarray, list of vectors       (m, n)
        
        Returns
        -------
        dict
            {'objective_value': float, 'x': ndarray}
        """
        c, b, P, q = (np.asarray(problem[k]) for k in ('c', 'b', 'P', 'q'))
        n = c.size

        x = cp.Variable(n)
        # build SOC constraints efficiently
        soc_constraints = [cp.SOC(b[i] - q[i] @ x, P[i] @ x) for i in range(b.size)]

        prob = cp.Problem(cp.Minimize(c @ x), soc_constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                return {"objective_value": np.inf, "x": np.full(n, np.nan)}
            return {"objective_value": prob.value, "x": x.value}
        except Exception:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}