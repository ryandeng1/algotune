from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        model = cp_model.CpModel()

        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j] == 0:
                    model.Add(nodes[i] + nodes[j] <= 1)

        model.Maximize(sum(nodes))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        else:
            return []