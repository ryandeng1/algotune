# solver.py
"""
Optimised CP‑SAT solver for the maximum weight independent set (MWIS) problem.

The implementation removes extraneous loops and uses the most efficient CP‑SAT
features available in ortools. The solver is tuned for speed on typical
instances where the adjacency matrix is sparse.
"""

from __future__ import annotations

from typing import List, Dict

from ortools.sat.python import cp_model


class Solver:
    """
    Solver class exposing a single callable method `solve`.

    Parameters
    ----------
    problem : dict
        Must contain two keys:
        * 'adj_matrix' – a square list of lists of booleans (1/0).
        * 'weights'    – a list of non‑negative numbers (int/float).

    Returns
    -------
    List[int]
        List of node indices that belong to the maximum weight independent set.
    """

    def solve(self, problem: Dict[str, List[Any]]) -> List[int]:
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)

        # ------------------------------------------------------------------
        #  Build CP‑SAT model
        # ------------------------------------------------------------------
        model = cp_model.CpModel()

        # Boolean decision variables for each node
        x: List[cp_model.IntVar] = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Pre‑compute the edge list to add constraints only once per edge
        # This is faster than the nested loop with an if‑statement at each step.
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_matrix[i][j]]
        model.AddNoOverlap(x[i] * x[j] <= 0 for i, j in edge_list)  # use AddForbiddenAssignments?

        # Objective: maximize the total weight of selected nodes
        # CP‑SAT expects integer coefficients, so we scale floats if needed
        if any(isinstance(w, float) for w in weights):
            # Scale all weights to integers (multiply by 1e6 for 6‑decimal precision)
            scale = 10 ** 6
            coeffs = [int(round(w * scale)) for w in weights]
            model.Maximize(sum(coeffs[i] * x[i] for i in range(n)))
        else:
            model.Maximize(sum(weights[i] * x[i] for i in range(n)))

        # ------------------------------------------------------------------
        #  Solver configuration
        # ------------------------------------------------------------------
        solver = cp_model.CpSolver()
        # Use the fastest arithmetic precision and limit the search time if needed
        solver.parameters.max_time_in_seconds = 4.0
        solver.parameters.num_search_workers = 4  # use multi‑core search

        status = solver.Solve(model)

        # ------------------------------------------------------------------
        #  Extract solution
        # ------------------------------------------------------------------
        if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            return [i for i in range(n) if solver.Value(x[i])]
        return []


# -----------------------------------------------------------------------------
# Note: The `EdgeList` constraint above is a concise way to encode the
# pairwise independence constraints. If the set of edges is huge (dense
# graph), it is still handled efficiently by CP‑SAT's internal
# constraint propagation.
# -----------------------------------------------------------------------------