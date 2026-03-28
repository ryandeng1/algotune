import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        f = np.array(problem["fixed_costs"])
        cap = np.array(problem["capacities"])
        dem = np.array(problem["demands"])
        tr = np.array(problem["transportation_costs"])

        n_fac = f.size
        n_cust = dem.size

        # Decision variables
        y = cp.Variable(n_fac, boolean=True)
        x = cp.Variable((n_fac, n_cust), boolean=True)

        # Objective
        obj = cp.Minimize(f @ y + cp.sum(cp.multiply(tr, x)))

        # Constraints
        constraints = []
        constraints.append(cp.sum(x, axis=0) == 1)                     # each customer served once
        constraints.append(cp.sum(x, axis=1) <= cap * y)               # capacity limits
        constraints.append(x <= y[:, None])                           # can't assign if facility closed

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False, warm_start=True)
        except Exception:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cust for _ in range(n_fac)],
            }

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cust for _ in range(n_fac)],
            }

        y_vals = np.rint(np.clip(y.value, 0, 1)).astype(bool)
        x_vals = np.rint(np.clip(x.value, 0, 1)).astype(float)

        return {
            "objective_value": float(prob.value),
            "facility_status": y_vals.tolist(),
            "assignments": x_vals.tolist(),
        }