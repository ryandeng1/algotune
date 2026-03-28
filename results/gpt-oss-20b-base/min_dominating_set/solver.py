from ortools.sat.python import cp_model
from typing import List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the minimum dominating set problem using OR‑Tools CP‑SAT.
        This implementation pre‑computes adjacency lists and removes
        unnecessary control flow to reduce model construction time.
        """
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency lists (including the node itself)
        neighbors = [
            [j for j in range(n) if problem[i][j] or i == j]
            for i in range(n)
        ]

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Each node must be dominated by itself or a neighbor
        for i in range(n):
            # Sum of BoolVars is at least 1
            model.AddBoolOr([x[j] for j in neighbors[i]])

        # Minimise the size of the dominating set
        model.Minimize(sum(x))

        # Create solver instance with a time limit (optional)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 0  # Use single thread for reproducibility
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        elif status == cp_model.FEASIBLE:
            # Return the best feasible solution found within the time limit
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        else:
            return []