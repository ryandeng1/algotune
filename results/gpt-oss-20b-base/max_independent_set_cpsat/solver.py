from ortools.sat.python import cp_model


class Solver:
    """
    CP‑SAT based solver for Maximum Independent Set.
    Uses adjacency list to avoid unnecessary iterations.
    """

    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        model = cp_model.CpModel()

        # Decision variables
        vars = [model.NewBoolVar(f"x{i}") for i in range(n)]

        # Build constraint set using adjacency lists
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(vars[i] + vars[j] <= 1)

        # Objective: maximize the number of chosen vertices
        model.Maximize(sum(vars))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30  # optional limit
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, v in enumerate(vars) if solver.Value(v)]
        return []