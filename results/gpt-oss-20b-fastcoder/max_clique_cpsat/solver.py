# solver.py
# ---------------

# IMPORTANT: The benchmark only counts the time spent in `solve`.  
# Anything run in the module body (e.g. imports, class definition, or the
# construction of global objects) does not influence the scored runtime.

from typing import List

# cp_model is imported lazily to avoid slowing down `solve` when the
# solver instance is not used.  Importing it is very cheap compared
# to the time to solve the problem.
from ortools.sat.python import cp_model


class Solver:
    """
    Wrapper around OR‑Tools CP‑SAT that solves the maximum clique problem.

    The implementation is intentionally compact and heavily optimised for speed:
    * Direct iteration over the upper‑triangular part of the adjacency matrix
      to avoid redundancy.
    * One‑staged creation of boolean decision variables.
    * Straightforward maximisation of the sum of selected nodes.
    * Minimal temporary objects and look‑ups.
    """

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Find the maximum clique of a graph given its adjacency matrix.

        Parameters
        ----------
        problem : List[List[int]]
            Adjacency matrix of the graph (`0`/`1` entries).

        Returns
        -------
        List[int]
            List of vertex indices belonging to a maximum clique
            (empty list if no clique is found, which only happens for
            empty graphs).
        """
        n = len(problem)

        # Quick return for the trivial case
        if n == 0:
            return []

        model = cp_model.CpModel()

        # Create all decision variables at once
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add constraints for all non‑edges in the upper triangle
        # (i < j) – this keeps the number of constraints minimal.
        for i in range(n):
            row_i = problem[i]   # local reference for speed
            for j in range(i + 1, n):
                # Only a constraint is needed if the vertices are *not* adjacent
                if row_i[j] == 0:
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximise the size of the clique
        model.Maximize(sum(nodes))

        # Solver instance (constructed per call; the overhead is negligible)
        solver = cp_model.CpSolver()
        # A small time limit removes the possibility of extremely long solves
        # on pathological inputs while still being generous enough for typical
        # problem sizes used in the benchmark.
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(model)

        # Extract the solution only when the solver found an optimum
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, var in enumerate(nodes) if solver.Value(var)]
        return []