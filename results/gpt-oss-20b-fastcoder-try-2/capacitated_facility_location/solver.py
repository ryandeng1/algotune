import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=int)
        capacities = np.asarray(problem["capacities"], dtype=int)
        demands = np.asarray(problem["demands"], dtype=int)
        trans_cost = np.asarray(problem["transportation_costs"], dtype=int)

        n_fac = fixed_costs.size
        n_cus = demands.size

        # Scaling 1e6 to keep values in int range
        SCALE = 1000000

        model = cp_model.CpModel()

        # ------------------------------
        # 1. Decision variables
        # ------------------------------
        y = [model.NewBoolVar(f"open_{i}") for i in range(n_fac)]
        # x[i][j] binary assignment variables
        x = [
            [model.NewBoolVar(f"x_{i}_{j}") for j in range(n_cus)]
            for i in range(n_fac)
        ]

        # ------------------------------
        # 2. Constraints
        # ------------------------------
        # Each customer is assigned exactly once
        for j in range(n_cus):
            model.AddExactlyOne(x[i][j] for i in range(n_fac))

        # Facility capacity constraints
        for i in range(n_fac):
            # sum_j demand[j] * x[i][j] <= capacities[i] * y[i]
            # All values are integers, we use LinearExpr
            expr = sum(demands[j] * x[i][j] for j in range(n_cus))
            model.Add(expr <= capacities[i] * y[i])

            # x[i][j] <= y[i] for all j (makes sense if facility closed)
            for j in range(n_cus):
                model.Add(x[i][j] <= y[i])

        # ------------------------------
        # 3. Objective
        # ------------------------------
        # scaled objective: fixed_cost + transport cost
        # use int scaling to avoid floats
        obj_expr = sum(fixed_costs[i] * y[i] for i in range(n_fac))
        obj_expr += sum(trans_cost[i][j] * x[i][j] for i in range(n_fac) for j in range(n_cus))
        model.Minimize(obj_expr)

        # ------------------------------
        # 4. Solve
        # ------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # give some time budget
        solver.parameters.num_search_workers = 0  # let the solver choose optimal threads

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.FEASIBLE_INACCURATE):
            # Infeasible / Unbounded / etc
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * n_cus for _ in range(n_fac)],
            }

        # ------------------------------
        # 5. Build result
        # ------------------------------
        facility_status = [bool(solver.Value(y[i])) for i in range(n_fac)]
        assignments = [[float(solver.Value(x[i][j])) for j in range(n_cus)] for i in range(n_fac)]
        objective_value = float(solver.ObjectiveValue())
        return {
            "objective_value": objective_value,
            "facility_status": facility_status,
            "assignments": assignments,
        }