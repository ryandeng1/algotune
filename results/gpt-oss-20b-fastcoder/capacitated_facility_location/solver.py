from typing import Any
import numpy as np
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Capacitated Facility Location Problem using OR-Tools CP‑SAT.

        Args:
            problem: Dictionary with keys:
                * fixed_costs: list/array of length n_facilities
                * capacities: list/array of length n_facilities
                * demands: list/array of length n_customers
                * transportation_costs: 2‑D list/array (n_facilities × n_customers)

        Returns:
            Dictionary with:
                - objective_value: optimal objective value
                - facility_status: list of bools for open facilities
                - assignments: matrix x_{ij} assignments (floats 0 or 1)
        """
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=int)
        capacities = np.asarray(problem["capacities"], dtype=int)
        demands = np.asarray(problem["demands"], dtype=int)
        trans_costs = np.asarray(problem["transportation_costs"], dtype=int)

        n_facilities, n_customers = fixed_costs.size, demands.size

        model = cp_model.CpModel()

        # Binary variables for opening facilities
        y = [model.NewBoolVar(f"y_{i}") for i in range(n_facilities)]

        # Binary variables for assignments
        x = [
            [model.NewBoolVar(f"x_{i}_{j}") for j in range(n_customers)]
            for i in range(n_facilities)
        ]

        # Each customer must be served by exactly one facility
        for j in range(n_customers):
            model.AddExactlyOne(x[i][j] for i in range(n_facilities))

        # Capacity constraints
        for i in range(n_facilities):
            model.Add(
                sum(demands[j] * x[i][j] for j in range(n_customers))
                <= capacities[i] * y[i]
            )
            # Linking x and y
            for j in range(n_customers):
                model.Add(x[i][j] <= y[i])

        # Objective
        obj = sum(fixed_costs[i] * y[i] for i in range(n_facilities))
        obj += sum(
            trans_costs[i][j] * x[i][j]
            for i in range(n_facilities)
            for j in range(n_customers)
        )
        model.Minimize(obj)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = 42

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        y_vals = [solver.Value(v) for v in y]
        x_vals = [[solver.Value(v) for v in row] for row in x]

        return {
            "objective_value": solver.ObjectiveValue(),
            "facility_status": [bool(v) for v in y_vals],
            "assignments": x_vals,
        }