from typing import List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Finds a maximum clique in the given adjacency matrix.
        """
        n = len(problem)
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f"x{i}") for i in range(n)]

        # add constraints only for non‑edges
        for i in range(n):
            pi = problem[i]
            xi = x[i]
            for j in range(i + 1, n):
                if pi[j] == 0:
                    m.Add(xi + x[j] <= 1)

        m.Maximize(sum(x))

        solver = cp_model.CpSolver()
        # a few extra parameters to speed up the search
        params = cp_model.CpSolverParameters()
        params.num_search_workers = 0  # single worker, less overhead
        solver.parameters.CopyFrom(params)

        status = solver.Solve(m)
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(x[i])]
        return []