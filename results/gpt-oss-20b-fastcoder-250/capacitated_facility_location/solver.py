from typing import Any, Dict, List, Tuple
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using OR-Tools CP-SAT.
        Returns a dictionary with the optimal objective value, facility status,
        and assignment matrix.
        """
        # Extract data
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=np.int64)
        capacities = np.asarray(problem["capacities"], dtype=np.int64)
        demands = np.asarray(problem["demands"], dtype=np.int64)
        transport = np.asarray(problem["transportation_costs"], dtype=np.int64)

        n_fac = fixed_costs.size
        n_cus = demands.size

        model = cp_model.CpModel()

        # Binary variables
        y = [model.NewBoolVar(f"y_{i}") for i in range(n_fac)]
        x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n_cus)]
             for i in range(n_fac)]

        # Customer must be served by exactly one facility
        for j in range(n_cus):
            model.Add(sum(x[i][j] for i in range(n_fac)) == 1)

        # Capacity constraints
        for i in range(n_fac):
            model.Add(
                sum(demands[j] * x[i][j] for j in range(n_cus)) <= capacities[i] * y[i]
            )

        # Link x and y
        for i in range(n_fac):
            for j in range(n_cus):
                model.Add(x[i][j] <= y[i])

        # Objective: fixed costs + transportation costs
        obj_terms = []
        for i in range(n_fac):
            obj_terms.append(fixed_costs[i] * y[i])
            for j in range(n_cus):
                obj_terms.append(transport[i][j] * x[i][j])
        model.Minimize(sum(obj_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("time_limit", 30)
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cus for _ in range(n_fac)],
            }

        # Extract solution
        fac_status = [bool(solver.Value(y[i])) for i in range(n_fac)]
        assign = [[float(solver.Value(x[i][j])) for j in range(n_cus)] for i in range(n_fac)]
        obj_val = solver.ObjectiveValue()

        return {
            "objective_value": obj_val,
            "facility_status": fac_status,
            "assignments": assign,
        }
