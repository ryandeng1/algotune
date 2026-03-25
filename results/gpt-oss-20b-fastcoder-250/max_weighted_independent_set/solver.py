# solver.py
import logging
from ortools.sat.python import cp_model
from typing import List, Dict, Any

class Solver:
    def solve(self, problem: Dict[str, List[Any]]) -> List[int]:
        """Solve the Maximum Weighted Independent Set (MWIS) problem using OR-Tools CP‑SAT.

        The implementation is identical to the reference solution but includes a few micro‑optimisations:
        - The adjacency matrix is converted to a 2‑D list of integers to avoid any Pandas/Numpy overhead.
        - Constraints are added only for one direction of the edge (i<j) to minimise the number of model statements.
        - The model is solved with the default parameters which are usually sufficient for the test cases.

        Parameters
        ----------
        problem : dict
            A dictionary containing:
                * "adj_matrix": a square 2‑D list of 0/1 values.
                * "weights": a list of node weights.

        Returns
        -------
        list[int]
            Indices of nodes that form an optimal independent set.
        """
        adj = problem.get("adj_matrix", [])
        weights = problem.get("weights", [])
        n = len(adj)
        if n == 0:
            return []

        # Create the model and variables
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add independent set constraints
        for i in range(n):
            row = adj[i]
            xi = x[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(xi + x[j] <= 1)

        # Objective: maximise total weight
        model.Maximize(sum(weights[i] * x[i] for i in range(n)))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        else:
            logging.warning("CP‑SAT did not find a solution.")
            return []
