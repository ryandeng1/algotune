from typing import List, Tuple
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        model = cp_model.CpModel()
        # Decision variables: x[i][p] = 1 if vertex i in A maps to vertex p in B
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # Each vertex in A can map to at most one vertex in B
        for i in range(n):
            model.Add(sum(x[i]) <= 1)

        # Each vertex in B can receive at most one vertex from A
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # Conflict constraints:
        # For every unordered pair of vertices (i,j) in A
        # and every ordered pair (p,q) of vertices in B
        # if A[i][j] == B[p][q] then we cannot map i→p and j→q simultaneously
        for i in range(n):
            for j in range(i + 1, n):
                aij = A[i][j]
                for p in range(m):
                    for q in range(m):
                        if aij == B[p][q]:
                            model.Add(x[i][p] + x[j][q] <= 1)

        # Objective: maximize the number of mapped vertices
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        return []