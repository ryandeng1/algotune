#!/usr/bin/env python3
# solver.py
# Optimised solver for matching maximum bipartite graph respecting adjacency mismatches.

from ortools.sat.python import cp_model
from typing import List, Tuple, Dict

class Solver:
    """
    Solve the maximum bipartite matching problem with the following constraints:
        * Each vertex from A can be matched to at most one vertex from B.
        * Each vertex from B can be matched to at most one vertex from A.
        * For any two vertices i, j in A and any two distinct vertices p, q in B,
          the pair can be matched only if A[i][j] == B[p][q] is false
          (i.e. mismatched adjacency forbids that pairing).
    """

    def solve(self, problem: Dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        model = cp_model.CpModel()

        # Decision variables
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # One-to-one constraints
        model.AddSumLessEquals([x[i][p] for p in range(m)], 1)   # each A‑vertex used at most once
        model.AddSumLessEquals([x[i][p] for i in range(n)], 1)   # each B‑vertex used at most once

        # Mismatch constraints
        for i in range(n):
            for j in range(i + 1, n):
                for p in range(m):
                    for q in range(m):
                        if p == q:
                            continue
                        if A[i][j] != B[p][q]:
                            model.Add(x[i][p] + x[j][q] <= 1)

        # Objective: maximize number of matched pairs
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120.0  # arbitrary time limit for robustness
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        return []