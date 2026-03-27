from typing import Any, List, Tuple
from ortools.sat.python import cp_model
from functools import lru_cache


class Solver:
    def solve(self, problem: dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        model = cp_model.CpModel()

        # Variables: x[i][p] = 1 if node i in G maps to node p in H
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # One‑to‑one mapping constraints
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # Pre‑compute the difference set: for each pair (i,j) store
        # which pairs (p,q) disagree in the adjacency matrices.
        @lru_cache(maxsize=None)
        def bad_edges(i: int, j: int) -> List[Tuple[int, int]]:
            return [(p, q) for p in range(m) for q in range(m)
                    if p != q and A[i][j] != B[p][q]]

        # Edge consistency constraints
        for i in range(n):
            for j in range(i + 1, n):
                bad = bad_edges(i, j)
                for p, q in bad:
                    model.Add(x[i][p] + x[j][q] <= 1)

        # Objective: maximize the size of the mapping
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        solver = cp_model.CpSolver()
        # Optional: limit the solver to use less memory
        solver.parameters.max_time_in_seconds = 60
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m)
                    if solver.Value(x[i][p]) == 1]
        return []