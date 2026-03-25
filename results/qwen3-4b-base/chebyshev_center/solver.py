import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        a_list = problem["a"]
        b_list = problem["b"]
        m = len(b_list)
        n = len(a_list[0])
        
        c = [np.linalg.norm(a_i) for a_i in a_list]
        
        solver = pywraplp.Solver.CreateSolver('LP')
        
        x_vars = [solver.NumVar(-solver.infinity(), solver.infinity(), f'x_{i}') for i in range(n)]
        r_var = solver.NumVar(0, solver.infinity(), 'r')
        
        for i in range(m):
            constraint = solver.Add(
                sum(a_list[i][j] * x_vars[j] for j in range(n)) + c[i] * r_var <= b_list[i]
            )
        
        solver.SetObjective(r_var, 1.0)
        solver.Solve()
        
        x_sol = [x_vars[i].solution_value() for i in range(n)]
        return {"solution": x_sol}
