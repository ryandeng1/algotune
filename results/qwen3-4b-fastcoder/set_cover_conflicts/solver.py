from typing import Any
from typing import NamedTuple
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]


class Solver:
    def solve(self, problem: Instance | tuple) -> list[int]:
        if not isinstance(problem, Instance):
            problem = Instance(*problem)
        n, sets, conflicts = problem
        
        coverage = [[] for _ in range(n)]
        for set_idx, s in enumerate(sets):
            for obj in s:
                coverage[obj].append(set_idx)
        
        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(len(sets))]
        
        for obj in range(n):
            model.Add(sum(set_vars[i] for i in coverage[obj]) >= 1)
        
        for conflict in conflicts:
            model.AddAtMostOne([set_vars[i] for i in conflict])
        
        model.Minimize(sum(set_vars))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [i for i in range(len(sets)) if solver.Value(set_vars[i]) == 1]
            return solution
        else:
            raise ValueError("No feasible solution found.")