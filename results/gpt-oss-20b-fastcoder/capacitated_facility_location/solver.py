from typing import Any
import numpy as np
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using OR-Tools CP-SAT solver.
        """
        # Convert data to numpy arrays for convenience
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=int)
        capacities = np.asarray(problem["capacities"], dtype=int)
        demands = np.asarray(problem["demands"], dtype=int)
        transportation_costs = np.asarray(problem["transportation_costs"], dtype=int)

        n_facilities = fixed_costs.size
        n_customers = demands.size

        model = cp_model.CpModel()

        # Boolean variables y[i] and x[i][j]
        y = [model.NewBoolVar(f"y_{i}") for i in range(n_facilities)]
        x = [
            [model.NewBoolVar(f"x_{i}_{j}") for j in range(n_customers)]
            for i in range(n_facilities)
        ]

        # Each customer must be assigned to exactly one facility
        for j in range(n_customers):
            model.AddExactlyOne(x[i][j] for i in range(n_facilities))

        # Capacity constraints and coupling between x and y
        for i in range(n_facilities):
            # If facility i is closed, no customers can be assigned
            model.Add(sum(demands[j] * x[i][j] for j in range(n_customers)) <= capacities[i] * y[i])
            for j in range(n_customers):
                # customer assignment implies facility open
                model.AddImplication(x[i][j], y[i])

        # Objective: minimize total cost
        total_cost = (
            sum(fixed_costs[i] * y[i] for i in range(n_facilities))
            + sum(
                transportation_costs[i][j] * x[i][j]
                for i in range(n_facilities)
                for j in range(n_customers)
            )
        )
        model.Minimize(total_cost)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # optional timeout
        solver.parameters.num_search_workers = 0  # use default parallelism
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Extract solution
        assign_matrix = []
        for i in range(n_facilities):
            row = []
            for j in range(n_customers):
                row.append(float(solver.Value(x[i][j])))
            assign_matrix.append(row)

        facility_status = [bool(solver.Value(y[i])) for i in range(n_facilities)]

        return {
            "objective_value": float(solver.ObjectiveValue()),
            "facility_status": facility_status,
            "assignments": assign_matrix,
        }