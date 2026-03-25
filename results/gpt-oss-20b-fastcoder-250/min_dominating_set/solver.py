# solver.py
import logging
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solve the Minimum Dominating Set problem for an undirected graph represented
        by its adjacency matrix.

        Parameters
        ----------
        problem : list[list[int]]
            Symmetric adjacency matrix of the graph.  problem[i][j] == 1
            indicates an edge between vertices i and j, 0 otherwise.

        Returns
        -------
        list[int]
            List of vertex indices that constitute a minimum dominating set.
        """
        n = len(problem)

        # Create the CP-SAT model
        model = cp_model.CpModel()

        # Boolean decision variable for each vertex: 1 if included in the dominating set
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add domination constraints: for every vertex, at least one of itself
        # or one of its neighbors must be selected.
        for i in range(n):
            neighbors = [x[i]]
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    neighbors.append(x[j])
            model.Add(sum(neighbors) >= 1)

        # Objective: minimize the number of selected vertices
        model.Minimize(sum(x))

        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5  # give a little timeout
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        else:
            # Should not happen for valid inputs ; log and return empty set
            logging.warning("No feasible solution found for the dominating set problem.")
            return []
