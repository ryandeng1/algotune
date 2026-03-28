from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        fixed_costs = np.array(problem["fixed_costs"])
        capacities = np.array(problem["capacities"])
        demands = np.array(problem["demands"])
        transportation_costs = np.array(problem["transportation_costs"])
        n_facilities = fixed_costs.size
        n_customers = demands.size

        y = cp.Variable(n_facilities, boolean=True)
        x = cp.Variable((n_facilities, n_customers), boolean=True)

        objective = cp.Minimize(fixed_costs @ y + cp.sum(cp.multiply(transportation_costs, x)))
        constraints = []
        for j in range(n_customers):
            constraints.append(cp.sum(x[:, j]) == 1)
        for i in range(n_facilities):
            constraints.append(demands @ x[i, :] <= capacities[i] * y[i])
            constraints.append(x[i, :] <= y[i])

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False)
        except Exception as e:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        y_rounded = np.clip(np.round(y.value), 0, 1)
        x_rounded = np.clip(np.round(x.value), 0, 1)

        facility_status = [bool(val) for val in y_rounded.tolist()]
        assignments = x_rounded.tolist()
        return {
            "objective_value": float(prob.value),
            "facility_status": facility_status,
            "assignments": assignments,
        }