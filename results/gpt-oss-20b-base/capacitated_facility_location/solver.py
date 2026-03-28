from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Extract data into NumPy arrays
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=float)
        capacities = np.asarray(problem["capacities"], dtype=float)
        demands = np.asarray(problem["demands"], dtype=float)
        transport_costs = np.asarray(problem["transportation_costs"], dtype=float)

        n_facilities = fixed_costs.size
        n_customers = demands.size

        # Decision variables
        y = cp.Variable(n_facilities, boolean=True)
        x = cp.Variable((n_facilities, n_customers), boolean=True)

        # Objective: facility opening + transportation cost
        objective = cp.Minimize(
            fixed_costs @ y + cp.sum(cp.multiply(transport_costs, x))
        )

        # Constraints
        constraints = []

        # Each customer served exactly once
        constraints.append(cp.sum(x, axis=0) == 1)

        # Facility capacity constraints
        constraints.append((x.T @ demands) <= capacities * y)

        # Assignments only to open facilities
        constraints.append(x <= y[:, None])

        # Build and solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False)
        except Exception:
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

        # Retrieve integer solutions
        y_vals = np.clip(np.rint(y.value), 0, 1).astype(int)
        x_vals = np.clip(np.rint(x.value), 0, 1).astype(int)

        return {
            "objective_value": float(prob.value),
            "facility_status": y_vals.astype(bool).tolist(),
            "assignments": x_vals.tolist(),
        }