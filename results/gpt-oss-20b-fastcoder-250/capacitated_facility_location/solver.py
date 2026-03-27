from typing import Any
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Optimized solver for the Capacitated Facility Location Problem.
        """
        # Convert data to NumPy arrays
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=float)
        capacities = np.asarray(problem["capacities"], dtype=float)
        demands = np.asarray(problem["demands"], dtype=float)
        transport_costs = np.asarray(problem["transportation_costs"], dtype=float)

        n_facilities = fixed_costs.shape[0]
        n_customers = demands.shape[0]

        # Decision variables
        y = cp.Variable(n_facilities, boolean=True)
        x = cp.Variable((n_facilities, n_customers), boolean=True)

        # Objective: facility opening + transport cost
        objective = cp.Minimize(
            fixed_costs @ y + cp.sum(cp.multiply(transport_costs, x))
        )

        # Constraints
        constraints = []

        # Each customer assigned to exactly one facility
        constraints.append(cp.sum(x, axis=0) == 1)

        # Demand must not exceed facility capacity when opened
        # ∑_j d_j * x_{ij} ≤ cap_i * y_i
        constraints.append(cp.multiply(demands, x).sum(axis=1) <= capacities * y)

        # A customer can be assigned to a facility only if it is open
        # x_{ij} ≤ y_i  for all i, j → x ≤ y[:, None]
        constraints.append(x <= cp.reshape(y, (-1, 1)))

        # Build and solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False)
        except Exception:  # pragma: no cover
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Check feasibility
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Extract and round solution (handle numerical tolerance)
        y_opt = np.clip(np.round(y.value), 0, 1)
        x_opt = np.clip(np.round(x.value), 0, 1)

        return {
            "objective_value": float(prob.value),
            "facility_status": bools := y_opt.astype(bool).tolist(),
            "assignments": x_opt.tolist(),
        }