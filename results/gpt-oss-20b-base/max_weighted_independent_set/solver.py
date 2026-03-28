from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, list]) -> list[int]:
        """Solve MWIS with CP‑SAT."""
        adj = problem['adj_matrix']
        w    = problem['weights']
        n    = len(adj)

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x{i}') for i in range(n)]

        # Only add constraints for existent edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    model.Add(x[i] + x[j] <= 1)

        model.Maximize(sum(w[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i, var in enumerate(x) if solver.Value(var)]
        return []