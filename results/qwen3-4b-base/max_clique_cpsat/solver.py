from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 0:
                    non_edges.append((i, j))
        
        for i, j in non_edges:
            model.Add(nodes[i] + nodes[j] <= 1)
        
        model.Maximize(sum(nodes))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []
