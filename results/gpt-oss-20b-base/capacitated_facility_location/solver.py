from typing import Any, Dict, List
import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    """
    Fast MIP solver for Capacitated Facility Location using OR-Tools CBC.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=np.float64)
        capacities = np.asarray(problem["capacities"], dtype=np.float64)
        demands = np.asarray(problem["demands"], dtype=np.float64)
        transport = np.asarray(problem["transportation_costs"], dtype=np.float64)

        n_facilities = fixed_costs.size
        n_customers = demands.size

        # Create solver
        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Decision variables
        y = [solver.IntVar(0, 1, f"y_{i}") for i in range(n_facilities)]
        x = [
            [solver.IntVar(0, 1, f"x_{i}_{j}") for j in range(n_customers)]
            for i in range(n_facilities)
        ]

        # Each customer must be assigned to exactly one facility
        for j in range(n_customers):
            solver.Add(solver.Sum(x[i][j] for i in range(n_facilities)) == 1)

        # Capacity constraints
        for i in range(n_facilities):
            solver.Add(
                solver.Sum(demands[j] * x[i][j] for j in range(n_customers))
                <= capacities[i] * y[i]
            )

        # Link constraints: cannot use a facility if it is closed
        for i in range(n_facilities):
            for j in range(n_customers):
                solver.Add(x[i][j] <= y[i])

        # Objective
        obj = solver.Sum(
            fixed_costs[i] * y[i] + solver.Sum(transport[i][j] * x[i][j] for j in range(n_customers))
            for i in range(n_facilities)
        )
        solver.Minimize(obj)

        # Solve
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Retrieve results
        facility_status: List[bool] = [bool(round(y[i].SolutionValue())) for i in range(n_facilities)]
        assignments: List[List[float]] = [
            [round(x[i][j].SolutionValue()) for j in range(n_customers)] for i in range(n_facilities)
        ]

        return {
            "objective_value": solver.Objective().Value(),
            "facility_status": facility_status,
            "assignments": assignments,
        }