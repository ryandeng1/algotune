from typing import Any, Dict
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solve a robust linear program with SOC constraints.
        The problem is:
            minimize   c^T x
            subject to ||P_i x||_2 <= b_i - q_i^T x   for i = 1..m

        Parameters
        ----------
        problem : dict
            Keys:
                - 'c' : (n,) array, cost vector
                - 'b' : (m,) array, RHS scalars
                - 'P' : (m, n, n) array, SPD matrices
                - 'q' : (m, n) array, vectors

        Returns
        -------
        dict
            - 'objective_value': optimal value (float)
            - 'x' : optimal primal vector (np.ndarray)
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)

        n = c.size
        m = b.size

        x = cp.Variable(n)

        # Build SOC constraints efficiently
        cons = []
        for i in range(m):
            # ||P_i x||_2 <= b_i - q_i^T x
            cons.append(cp.SOC(b[i] - q[i] @ x, P[i] @ x))

        objective = cp.Minimize(c @ x)
        prob = cp.Problem(objective, cons)

        try:
            # ECOS handles SOC and is usually faster than CLARABEL
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=1000)
        except Exception:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        return {"objective_value": float(prob.value), "x": x.value}