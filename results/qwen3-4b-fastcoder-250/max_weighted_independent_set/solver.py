from ortools.sat.python import cp_model
import os

class Solver:
    def solve(self, problem: dict[str, list]) -> list[int]:
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 1:
                    edges.append((i, j))
        
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        for i, j in edges:
            model.Add(nodes[i] + nodes[j] <= 1)
        
        model.Maximize(sum(weights[i] * nodes[i] for i in range(n)))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_threads = os.cpu_count()
        
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i])]
        else:
            return []
