import math
from typing import List
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """Solve the minimum dominating set with a CP‑SAT model.

        Build an adjacency list first to avoid repeated scan of the
        adjacency matrix.  For each vertex create a boolean variable and
        add a single linear constraint that its own variable plus those
        of its neighbours must be at least one.
        """
        n = len(problem)

        # Pre‑compute neighbors for each vertex
        neighbors_of: List[List[int]] = [None] * n
        for i in range(n):
            neigh = [i]
            row = problem[i]
            for j, val in enumerate(row):
                if val == 1:
                    neigh.append(j)
            neighbors_of[i] = neigh

        # CP‑SAT model
        model = cp_model.CpModel()
        vertices = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add domination constraints
        for i in range(n):
            neigh_vars = [vertices[j] for j in neighbors_of[i]]
            model.Add(sum(neigh_vars) >= 1)

        # Minimise the size of the set
        model.Minimize(sum(vertices))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # optional time limit
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(vertices[i])]
        return []