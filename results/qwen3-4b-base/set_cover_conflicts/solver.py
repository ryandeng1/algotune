import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: tuple) -> list[int]:
        n, sets, conflicts = problem
        cover = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                if obj < n:
                    cover[obj].append(i)
        
        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(len(sets))]
        
        for obj in range(n):
            model.Add(sum(set_vars[i] for i in cover[obj]) >= 1)
        
        for conflict in conflicts:
            model.AddAtMostOne([set_vars[i] for i in conflict])
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [i for i in range(len(sets)) if solver.Value(set_vars[i]) == 1]
            return solution
        else:
            raise ValueError("No feasible solution found.")
