# solver.py
from ortools.sat.python import cp_model
from typing import List

class Solver:
    """
    A fast implementation of the maximum independent set solver based on OR‑Tools CP‑SAT.
    """

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the maximum independent set problem for an undirected graph given as an
        adjacency matrix.

        Parameters
        ----------
        problem : List[List[int]]
            2‑D adjacency matrix (0/1) of the graph.  The graph is assumed to be
            simple, undirected and without self‑loops.

        Returns
        -------
        List[int]
            Indices of the nodes that form a maximum independent set.
            If the problem is infeasible an empty list is returned.
        """
        n = len(problem)

        # Create Boolean variables: x[i] = 1 if node i is selected.
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x[{i}]") for i in range(n)]

        # Collect all edges once, then add the constraints in a tight loop.
        # This is considerably faster than evaluating the matrix element per
        # constraint construction because the Python interpreter has to
        # execute fewer operations and the CP‑SAT API is only called once per
        # edge.
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)

        # Objective: maximise the number of selected nodes.
        model.Maximize(sum(x))

        # Resolve.
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return []