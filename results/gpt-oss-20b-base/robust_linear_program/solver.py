import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, int]:
        """Solve a robust LP with SOC constraints as fast as possible."""
        # unpack problem data
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)

        m, n = len(b), c.size
        x = cp.Variable(n)

        # build constraints once
        constraints = []
        for i in range(m):
            constraints.append(cp.SOC(b[i] - q[i] @ x, P[i] @ x))

        # formulate and solve
        prob = cp.Problem(cp.Minimize(c @ x), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                return {"objective_value": np.inf, "x": np.full(n, np.nan)}
            return {"objective_value": prob.value, "x": x.value}
        except Exception:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}