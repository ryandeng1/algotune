import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        fixed_costs = np.array(problem["fixed_costs"])
        capacities = np.array(problem["capacities"])
        demands = np.array(problem["demands"])
        transportation_costs = np.array(problem["transportation_costs"])
        n_facilities = fixed_costs.size
        n_customers = demands.size
        
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }
        
        y_vars = [solver.IntVar(0, 1, f'y_{i}') for i in range(n_facilities)]
        x_vars = [[solver.IntVar(0, 1, f'x_{i}_{j}') for j in range(n_customers)] for i in range(n_facilities)]
        
        for j in range(n_customers):
            total = 0
            for i in range(n_facilities):
                total += x_vars[i][j]
            solver.Add(total == 1)
        
        for i in range(n_facilities):
            total_demand = 0.0
            for j in range(n_customers):
                total_demand += demands[j] * x_vars[i][j]
            solver.Add(total_demand <= capacities[i] * y_vars[i])
        
        for i in range(n_facilities):
            for j in range(n_customers):
                solver.Add(x_vars[i][j] <= y_vars[i])
        
        objective = solver.Objective()
        for i in range(n_facilities):
            objective.SetCoefficient(y_vars[i], fixed_costs[i])
        for i in range(n_facilities):
            for j in range(n_customers):
                objective.SetCoefficient(x_vars[i][j], transportation_costs[i][j])
        
        objective.SetMinimization()
        
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }
        
        facility_status = [int(y_vars[i].solution_value()) for i in range(n_facilities)]
        assignments = [[int(x_vars[i][j].solution_value()) for j in range(n_customers)] for i in range(n_facilities)]
        
        return {
            "objective_value": float(solver.Objective().Value()),
            "facility_status": facility_status,
            "assignments": assignments,
        }
