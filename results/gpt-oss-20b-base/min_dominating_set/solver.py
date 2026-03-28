from typing import List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the Minimum Dominating Set (MDS) problem using OR‑Tools CP‑SAT.
        The implementation is deliberately concise to keep the CP‑SAT model small
        and therefore fast.
        """
        n = len(problem)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(n)]

        # For each vertex i, we need at least one selected vertex among
        # i itself and all of its neighbors.
        for i in range(n):
            # sum of Booleans equals number of selected vertices in the closed neighborhood
            neighborhood = [x[i]]
            # Build the neighborhood list in a tight loop
            row = problem[i]
            for j, has_edge in enumerate(row):
                if has_edge:
                    neighborhood.append(x[j])
            model.Add(sum(neighborhood) >= 1)

        # Objective: minimize the number of selected vertices
        model.Minimize(sum(x))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # optional time limit
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, var in enumerate(x) if solver.Value(var) == 1]
        else:
            return []