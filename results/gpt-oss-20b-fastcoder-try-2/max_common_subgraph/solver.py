from typing import List, Tuple
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A, B = problem["A"], problem["B"]
        n, m = len(A), len(B)

        # Create the CP-SAT model
        model = cp_model.CpModel()

        # Decision variables: x[i][p] == 1 iff vertex i of A is mapped to vertex p of B
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # Each vertex of A maps to at most one vertex of B
        for i in range(n):
            model.Add(sum(x[i]) <= 1)

        # Each vertex of B receives at most one vertex of A
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # Incompatibility constraints:
        # If A[i][j] != B[p][q] and i < j, p < q, then they cannot both be mapped
        # We add only the necessary constraints (i<j, p<q) to avoid duplication
        for i in range(n):
            ai = A[i]
            for j in range(i + 1, n):
                aij = ai[j]
                bj = B[j]
                for p in range(m):
                    bp = B[p]
                    for q in range(p + 1, m):
                        if aij != bp[q]:
                            model.Add(x[i][p] + x[j][q] <= 1)
                        if aij != B[q][p]:
                            model.Add(x[i][q] + x[j][p] <= 1)

        # Objective: maximize the number of matched vertices
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0   # optional timeout
        status = solver.Solve(model)

        # Extract the mapping
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p])]
        return []