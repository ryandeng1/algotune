from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Precompute adjacency list for efficiency
        adj = [[j for j in range(n) if problem[i][j] == 1] for i in range(n)]
        
        # Add domination constraints efficiently
        for i in range(n):
            model.Add(sum([nodes[i]] + [nodes[j] for j in adj[i]]) >= 1)
        
        # Minimize the size of the dominating set
        model.Minimize(sum(nodes))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        else:
            return []