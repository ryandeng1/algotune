from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        model = cp_model.CpModel()

        # x[i][p] = 1 if node i in G is mapped to node p in H
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # One‑to‑one mapping constraints
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # Edge consistency constraints (cover all p != q)
        A_local = A
        B_local = B
        x_local = x
        for i in range(n):
            for j in range(i + 1, n):
                a_ij = A_local[i][j]
                for p in range(m):
                    for q in range(m):
                        if p != q and B_local[p][q] != a_ij:
                            model.Add(x_local[i][p] + x_local[j][q] <= 1)

        # Objective: maximize size of the mapping
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        else:
            return []