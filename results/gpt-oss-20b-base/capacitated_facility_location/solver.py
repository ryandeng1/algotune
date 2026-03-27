from typing import Any
import pulp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using PuLP with CBC solver.

        Args:
            problem: A dictionary containing problem parameters.

        Returns:
            A dictionary containing:
                - objective_value: optimal objective value
                - facility_status: list of bools for open facilities
                - assignments: matrix x_{ij} assignments
        """
        # Extract data
        fixed_costs = np.array(problem["fixed_costs"])
        capacities = np.array(problem["capacities"])
        demands = np.array(problem["demands"])
        transportation_costs = np.array(problem["transportation_costs"])

        n_facilities = len(fixed_costs)
        n_customers = len(demands)

        # Create problem instance
        model = pulp.LpProblem("Capacitated_Facility_Location", pulp.LpMinimize)

        # Decision variables
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_facilities)]
        x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n_customers)]
             for i in range(n_facilities)]

        # Objective: fixed costs + transportation costs
        model += (pulp.lpSum(fixed_costs[i] * y[i] for i in range(n_facilities)) +
                  pulp.lpSum(transportation_costs[i][j] * x[i][j]
                             for i in range(n_facilities)
                             for j in range(n_customers)))

        # Constraints
        # Each customer must be assigned to exactly one facility
        for j in range(n_customers):
            model += pulp.lpSum(x[i][j] for i in range(n_facilities)) == 1

        # Demand & capacity constraints
        for i in range(n_facilities):
            # Total demand assigned to facility i must not exceed its capacity if it is open
            model += pulp.lpSum(demands[j] * x[i][j] for j in range(n_customers)) <= capacities[i] * y[i]
            # Assignment only allowed if facility is open
            for j in range(n_customers):
                model += x[i][j] <= y[i]

        # Solve
        try:
            model.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        status = pulp.LpStatus.get(model.status, "")
        if status not in ("Optimal", "Optimal (integer)", "Optimal (within tolerance)"):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Retrieve solution
        y_vals = np.array([bool(pulp.value(var)) for var in y], dtype=int)
        x_vals = np.array([[bool(pulp.value(var)) for var in row] for row in x], dtype=int)

        facility_status = [bool(val) for val in y_vals.tolist()]
        assignments = x_vals.tolist()

        return {
            "objective_value": float(pulp.value(model.objective)),
            "facility_status": facility_status,
            "assignments": assignments,
        }