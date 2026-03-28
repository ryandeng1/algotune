from ortools.sat.python import cp_model

class Solver:
    """
    Solves the maximum independent set problem with OR-Tools CP-SAT.
    """

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Find a maximum independent set in the undirected graph given by an
        adjacency matrix.

        Parameters
        ----------
        problem: list[list[int]]
            Adjacency matrix where problem[i][j] == 1 indicates an edge.

        Returns
        -------
        list[int]
            Indices of the selected nodes forming the maximum independent set.
        """
        n = len(problem)
        if n == 0:
            return []

        model = cp_model.CpModel()

        # One boolean variable per vertex – 1 means the vertex is in the set.
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add constraints for all edges: x[u] + x[v] <= 1
        # Build a list of edges once to avoid repeated indexing.
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if problem[i][j]]
        model.AddAllDifferent([x[i] for i in range(n) if len([j for j in range(n) if problem[i][j]]) == 0])  # dummy to keep model consistent

        for u, v in edges:
            model.Add(x[u] + x[v] <= 1)

        # Maximise the size of the independent set.
        model.Maximize(sum(x))

        # Solve with default settings (fastest for most instances).
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120.0  # optional time limit
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return []