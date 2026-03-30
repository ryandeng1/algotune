# solver.py

from ortools.sat.python import cp_model


class Solver:
    """
    Optimised CP‑SAT solver for the maximum clique problem.
    """

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Find a maximum clique of the graph described by the adjacency matrix.

        Parameters
        ----------
        problem : List[List[int]]
            Adjacency matrix of the graph.  `problem[i][j]` is 1 iff there is an
            edge {i, j}.

        Returns
        -------
        List[int]
            List of node indices that form a maximum clique.  The list is
            empty if the solver cannot find an optimum (should not happen
            for small graphs).
        """
        n = len(problem)
        model = cp_model.CpModel()
        # Boolean variables: xi = 1 iff node i is chosen
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add constraints: two non‑adjacent nodes cannot both be chosen
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j] == 0:               # not connected
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Maximise the total number of chosen nodes
        model.Maximize(sum(nodes))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            # Extract the chosen vertices
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return []