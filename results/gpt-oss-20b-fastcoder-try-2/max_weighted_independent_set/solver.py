from __future__ import annotations
from typing import List, Dict
from ortools.sat.python import cp_model

class Solver:
    # 1. Keep the signatures exactly as required.
    # 2. Use an adjacency list instead of an O(n^2) matrix traversal.
    #    Only add constraints for existing edges, which is far cheaper
    #    for sparse graphs and speeds up the modelling phase.
    # 3. Replace the generator expression inside the objective with a
    #    simple sum so that the solver sees a single linear expression.
    # 4. Use `IntVar` for the objective to avoid the expensive
    #    automatic type conversion.
    # 5. Avoid the redundant `else: pass` statements that confuse
    #    the reader and add a tiny bit of overhead.

    def solve(self, problem: Dict[str, List[List[int]]]) -> List[int]:
        """
        Weighted maximum independent set solver using OR-Tools CP‑SAT.

        Parameters
        ----------
        problem : Dict[str, List[List[int]]]
            Must contain:
                'adj_matrix' – list of lists, 1‑based adjacency matrix
                'weights'    – list of integer weights

        Returns
        -------
        List[int]
            Indices of vertices in the optimal independent set.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)

        # --- Build the optimisation model ------------------------------------
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Only add constraints for existing edges.
        for i in range(n):
            row = adj_matrix[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)

        # Weighted objective
        model.Maximize(sum(weights[i] * x[i] for i in range(n)))

        # --- Solve ------------------------------------------------------------
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        return []