import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, object]:
        """Solve the robust LP using CVXPY.

        Parameters
        ----------
        problem : dict
            Keys:
                * ``c`` : 1‑D array, objective coefficients
                * ``b`` : 1‑D array, RHS of constraints
                * ``P`` : list/array of PSD matrices
                * ``q`` : list/array of vectors
        Returns
        -------
        dict
            ``objective_value`` : optimal objective (float)
            ``x`` : optimal decision variable (1‑D array)
        """
        c = np.asarray(problem["c"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)

        n = c.size
        m = P.shape[0]

        x = cp.Variable(n)

        cons = [
            cp.SOC(b[i] - q[i] @ x, P[i] @ x) for i in range(m)
        ]

        prob = cp.Problem(cp.Minimize(c @ x), cons)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}
        return {"objective_value": float(prob.value), "x": x.value}