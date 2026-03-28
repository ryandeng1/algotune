from ortools.sat.python import cp_model
from typing import List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """Finds a minimum dominating set of a graph given by an adjacency matrix."""
        n = len(problem)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Pre‑compute neighbor lists to avoid repeated iteration inside the
        # constraint building loop.
        neigh = [[] for _ in range(n)]
        for i in range(n):
            row = problem[i]
            lst = [x[i]]  # a node dominates itself
            for j, val in enumerate(row):
                if val:
                    lst.append(x[j])
            neigh[i] = lst

        # One constraint per node: at least one node in its closed neighborhood
        add = model.Add
        for i in range(n):
            add(sum(neigh[i]) >= 1)

        # Objective: minimise the size of the dominating set
        model.Minimize(sum(x))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(x[i])]
        return []