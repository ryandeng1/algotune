from ortools.linear_solver import pywraplp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        a = problem["a"]
        b = problem["b"]
        n = len(a[0])
        m = len(b)
        
        c = [np.linalg.norm(a_i) for a_i in a]
        
        solver = pywraplp.Solver.CreateSolver('LP')
        
        x_vars = [solver.NumVar(-1e10, 1e10, f'x_{i}') for i in range(n)]
        r_var = solver.NumVar(0, 1e10, 'r')
        
        for i in range(m):
            constraint = solver.Constraint(-1e10, b[i])
            for j in range(n):
                constraint.SetCoefficient(x_vars[j], a[i][j])
            constraint.SetCoefficient(r_var, c[i])
        
        solver.SetObjective(-r_var)
        solver.Solve()
        
        return {"solution": [x_vars[i].solution_value() for i in range(n)]}
