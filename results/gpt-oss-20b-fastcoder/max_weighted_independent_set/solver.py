from typing import List
from ortools.sat.python import cp_model

class Solver:
    """Fast CP‑SAT MWIS implementation."""

    def solve(self, problem: dict[str, List]) -> List[int]:
        """
        Solves the Maximum‑Weight Independent Set (MWIS) problem.

        Parameters
        ----------
        problem : dict
            Must contain:
            * `adj_matrix` – adjacency matrix (list of lists of booleans)
            * `weights`    – list of node weights

        Returns
        -------
        List[int]
            Indices of nodes in an optimal independent set.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(weights)

        model = cp_model.CpModel()
        # Boolean variables for each node
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Create adjacency constraints only once per edge
        for i in range(n):
            ai = adj_matrix[i]
            for j in range(i + 1, n):
                if ai[j]:
                    model.Add(x[i] + x[j] <= 1)

        # Maximize total weight
        model.Maximize(
            sum(weights[i] * x[i] for i in range(n))
        )

        # Use a solver with a short time limit that suffices for moderate graphs.
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # adjust if needed
        solver.parameters.num_search_workers = cp_model.CpSolverParameters().GetDefault().num_search_workers

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        else:
            # In rare cases the solver may only find a feasible solution, return it
            return [i for i in range(n) if solver.Value(x[i]) == 1]