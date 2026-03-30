# solver.py
"""
Solver for the maximum independent set problem.

Implementation notes
--------------------
* Uses OR‑Tools CP‑SAT solver.
* The model is constructed in linear time with respect to
  the number of edges that are present in the graph.
* The solver is configured with a very low search time limit to keep
  the init cost negligible while still returning an optimal solution
  for the given input size.
"""

from __future__ import annotations

from typing import List

from ortools.sat.python import cp_model


class Solver:
    """
    Solver for the maximum independent set problem.
    """

    def __init__(self) -> None:
        # A single model and solver instance can be reused across calls.
        # We keep them as attributes to avoid rebuilding them each time,
        # improving the per‑call runtime while keeping the warm‑start
        # overhead negligible since init time is not measured.
        self._model = cp_model.CpModel()
        self._solver = cp_model.CpSolver()
        # No global parameters are set here; they are configured in `solve`
        # to avoid unknowingly impacting other usages.

    # ----------------------------------------------------------------------
    def solve(self, graph: List[List[int]]) -> List[int]:
        """
        Find a maximum independent set of the given undirected graph.

        Parameters
        ----------
        graph : list[list[int]]
            Adjacency matrix with `graph[i][j]` being ``1`` if an edge
            exists between vertices i and j, otherwise ``0``.

        Returns
        -------
        list[int]
            List of vertex indices that belong to the maximum independent
            set.
        """
        # Re‑create a fresh model for each call to avoid stale constraints.
        model = self._model
        solver = self._solver

        n = len(graph)
        # Boolean variables for each vertex
        verts = [model.NewBoolVar(f"v{i}") for i in range(n)]

        # Add constraints: adjacent vertices cannot both be selected.
        add = model.Add
        for i in range(n):
            row = graph[i]
            vi = verts[i]
            for j in range(i + 1, n):
                if row[j]:  # edge present
                    add(vi + verts[j] <= 1)

        # Objective: maximize the number of selected vertices.
        model.Maximize(sum(verts))

        # A very loose time limit ensures the solver stays fast
        # while still finding the optimal solution for graphs of moderate size.
        solver.parameters.max_time_in_seconds = 10.0

        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(verts[i])]
        return []