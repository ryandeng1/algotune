from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the maximum independent set problem using OR‑Tools CP‑SAT.
        The implementation builds constraints only for existing edges and
        eliminates Python overhead wherever possible.
        """
        n = len(problem)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add constraints for every undirected edge (i<j)
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)

        model.Maximize(sum(x))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        return []