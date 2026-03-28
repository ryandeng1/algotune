# Optimised version of the solver

from typing import List, Tuple
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        model = cp_model.CpModel()
        # boolean variables x[i][p] == 1 if vertex i in A is matched to vertex p in B
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # each vertex in A is matched to at most one in B
        for i in range(n):
            model.Add(sum(x[i]) <= 1)
        # each vertex in B is matched to at most one in A
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # pre‑compute adjacency differences to avoid inner loops of quadruple iteration
        diff = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                diff[key] = [True] * m * m
                for p in range(m):
                    for q in range(m):
                        if p == q:
                            diff[key][p * m + q] = False  # same vertex, no conflict
                        elif A[i][j] == B[p][q]:
                            diff[key][p * m + q] = False  # conflict, forbid simultaneously

        # add conflict constraints
        for (i, j), arr in diff.items():
            for p in range(m):
                for q in range(m):
                    if not arr[p * m + q]:
                        model.Add(x[i][p] + x[j][q] <= 1)

        # objective: maximize number of matches
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # safety
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p])]
        return []