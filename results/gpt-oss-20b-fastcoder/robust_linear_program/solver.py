import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, object]:
        c = np.asarray(problem["c"])
        b = np.asarray(problem["b"])
        P = np.asarray(problem["P"])
        q = np.asarray(problem["q"])
        n = c.size

        x = cp.Variable(n)
        cons = [cp.SOC(b[i] - q[i] @ x, P[i].T @ x) for i in range(len(P))]

        cp_prob = cp.Problem(cp.Minimize(c @ x), cons)
        try:
            cp_prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}

        if cp_prob.status not in {"optimal", "optimal_inaccurate"}:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}

        return {"objective_value": cp_prob.value, "x": x.value}