import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        adj = np.array(problem)
        neighbors = [np.where(adj[i] == 1)[0].tolist() for i in range(n)]
        
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        for i in range(n):
            model.Add(nodes[i] + sum(nodes[j] for j in neighbors[i]) >= 1)
        
        model.Minimize(sum(nodes))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        else:
            return []
