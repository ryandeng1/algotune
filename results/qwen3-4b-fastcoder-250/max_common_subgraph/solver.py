import random
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)
        model = cp_model.CpModel()
        
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]
        
        for i in range(n):
            model.Add(sum(x[i]) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)
        
        for i in range(n):
            for j in range(i + 1, n):
                a_ij = A[i][j]
                for p in range(m):
                    for q in range(m):
                        if p == q:
                            continue
                        if B[p][q] != a_ij:
                            model.Add(x[i][p] + x[j][q] <= 1)
        
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        else:
            return []
