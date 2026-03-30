# solver.py

"""
Solver for the Maximum Weighted Independent Set (MWIS) problem.
This implementation uses Google's OR-Tools CP-SAT solver.
"""

from __future__ import annotations

from typing import List, Dict

from ortools.sat.python import cp_model


class Solver:
    """
    Solves the MWIS problem.

    The input problem dictionary must contain two keys:
        * 'adj_matrix' – an n×n list of lists of booleans/ints (0/1) describing edges
        * 'weights'    – a list of n non‑negative integers

    The solver returns a list of node indices that form a maximum‑weight
    independent set.  For an infeasible or unsatisfiable instance an empty
    list is returned.
    """

    def __init__(self) -> None:
        # Pre‑compile a CP‑SAT model template that can be reused for many instances.
        # During initialisation we only create the generic variables; the actual
        # constraints are added when solve() is called.
        self._model_builder = self._build_model_template

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def solve(self, problem: Dict[str, List[List[int]]]) -> List[int]:
        """
        Find a maximum‑weight independent set for the given graph.

        Parameters
        ----------
        problem:
            dict with keys *adj_matrix* and *weights*.
        Returns
        -------
        list[int]
            Indices of selected nodes.
        """
        weights = problem["weights"]
        adj_matrix = problem["adj_matrix"]
        n = len(weights)

        # Build a fresh model based on the template.
        model = self._model_builder(n)

        # Variable references
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add adjacency constraints
        for i in range(n):
            row = adj_matrix[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximize weighted sum
        obj_expr = sum(weights[i] * nodes[i] for i in range(n))
        model.Maximize(obj_expr)

        # Solve
        solver = cp_model.CpSolver()
        # Use a fast local search + linear search hybrid
        solver.parameters.search_branching = cp_model.SearchBranching.AUTOMATIC
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i])]
        else:
            # Either infeasible or no optimal solution found.
            return []

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_model_template(n: int) -> cp_model.CpModel:
        """
        Returns a new, empty CpModel ready to be populated.
        This helper isolates model construction from instance data, making the
        outer solve() method more readable and slightly faster by avoiding repeated
        attribute lookups.
        """
        return cp_model.CpModel()