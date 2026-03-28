from ortools.sat.python import cp_model
from typing import List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the maximum independent set problem using OR‑Tools CP‑SAT.

        Parameters
        ----------
        problem : List[List[int]]
            Adjacency matrix of the graph (0/1 symmetric).

        Returns
        -------
        List[int]
            The vertex indices of a maximum independent set.
        """
        n = len(problem)
        if n == 0:
            return []

        # Create solver model
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add edge constraints. Even for dense graphs this is linear time in the
        # number of edges because we iterate only over the upper triangle.
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximise the number of selected vertices
        model.Maximize(sum(nodes))

        # Solve
        solver = cp_model.CpSolver()
        # Limit search time for very large graphs (optional, remove if not needed)
        # solver.parameters.max_time_in_seconds = 120.0

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract solution
            return [i for i in range(n) if solver.Value(nodes[i])]
        return []