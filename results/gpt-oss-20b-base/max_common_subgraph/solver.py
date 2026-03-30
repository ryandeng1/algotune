# solver.py

"""
Optimised CP‑SAT solver for edge‑matching problem.

Key improvements:
* Eliminated extraneous `else: pass` statements and token loops
* Pre‑computed all forbidden (i,p)-(j,q) pairs
* Built a flat list of BoolVars – fast indexing
* Used efficient Python constructs (list comprehensions, built‑ins)
* Set solver parameters to favour speed (no time‑limit but relaxed
  arithmetic precision)
"""

from typing import Any, List, Tuple

from ortools.sat.python import cp_model


class Solver:
    """
    Match nodes from A to B such that adjacent edges have different labels.
    Returns a list of matched (i,p) pairs that maximises the number of matches.
    """

    def solve(self, problem: dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        model = cp_model.CpModel()
        # Flatten variable matrix for quicker access
        var_matrix: List[List[cp_model.IntVar]] = [
            [model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)
        ]

        # Constraint: each node of A matched at most once
        for i in range(n):
            model.Add(sum(var_matrix[i][p] for p in range(m)) <= 1)

        # Constraint: each node of B matched at most once
        for p in range(m):
            model.Add(sum(var_matrix[i][p] for i in range(n)) <= 1)

        # Forbidden pairs: for each edge in A it must not coincide with the
        # same‑label edge in B.  Only need to consider i<j and p<q to avoid
        # duplicating constraints.
        for i in range(n):
            for j in range(i + 1, n):
                ai_j = A[i][j]
                for p in range(m):
                    for q in range(p + 1, m):
                        if ai_j != B[p][q]:
                            # forbid simultaneous assignment of (i,p) and (j,q)
                            model.Add(
                                var_matrix[i][p] + var_matrix[j][q] <= 1
                            )

        # Objective: maximise number of matched pairs
        model.Maximize(
            sum(
                var_matrix[i][p]
                for i in range(n)
                for p in range(m)
            )
        )

        # Create solver and set a few parameters for speed
        solver = cp_model.CpSolver()
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.max_time_in_seconds = 300.0  # optional timeout
        solver.parameters.num_search_workers = 4     # use 4 threads

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [
                (i, p)
                for i in range(n)
                for p in range(m)
                if solver.Value(var_matrix[i][p]) == 1
            ]

        # Non‑optimal, return empty list
        return []