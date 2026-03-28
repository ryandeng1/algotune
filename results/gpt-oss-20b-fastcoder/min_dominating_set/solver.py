from ortools.sat.python import cp_model
from typing import List

class Solver:
    @staticmethod
    def solve(problem: List[List[int]]) -> List[int]:
        """
        Minimum dominating set using OR-Tools CP-SAT.

        The constraint for each node i is that at least one of its
        incident vertices (including itself) is selected.  This way
        every node either belongs to the set or has a selected neighbour.

        The objective is simply to minimise the number of selected vertices.
        """
        n = len(problem)
        model = cp_model.CpModel()
        # Boolean variable for each node: 1 if selected, 0 otherwise
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # For each node, at least one of its neighbours (or itself) must be selected.
        for i, row in enumerate(problem):
            # Build the list of variables that cover node i
            cover_vars = [x[j] for j, val in enumerate(row) if val == 1]
            cover_vars.append(x[i])          # a node covers itself
            model.AddBoolOr(cover_vars)

        # Objective: minimise sum of selected nodes
        model.Minimize(sum(x))

        solver = cp_model.CpSolver()
        # Increase the time limit for larger instances (adjust if necessary)
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        return []