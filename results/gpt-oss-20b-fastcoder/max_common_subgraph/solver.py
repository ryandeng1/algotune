from ortools.sat.python import cp_model
from typing import List, Tuple, Dict

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)

        # create CP-SAT model
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # each vertex in A and B used at most once
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # pairwise conflicts: if A[i][j] != B[p][q] then cannot choose i-p and j-q simultaneously
        for i in range(n):
            for j in range(i + 1, n):
                aij = A[i][j]
                for p in range(m):
                    for q in range(m):
                        if p == q:
                            continue
                        if aij != B[p][q]:
                            model.Add(x[i][p] + x[j][q] <= 1)

        # objective: maximize number of matched vertices
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        # solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # tune if needed
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m)
                    if solver.Value(x[i][p]) == 1]
        return []