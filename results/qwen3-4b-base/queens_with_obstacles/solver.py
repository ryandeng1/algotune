class Solver:
    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        from ortools.sat.python import cp_model
        import numpy as np
        from typing import Any
        from collections.abc import Iterator

        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()

        # Decision variables
        queens = [[model.NewBoolVar(f"queen_{r}_{c}") for c in range(m)] for r in range(n)]

        # Constraint: No queens on obstacles
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)

        # Precompute reach positions for all cells
        reach_positions = [[[] for _ in range(m)] for _ in range(n)]
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_list = list(queen_reach(instance, (r, c)))
                    reach_positions[r][c] = reach_list

        # Constraint: No two queens attack each other
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    model.Add(
                        sum(queens[nr][nc] for nr, nc in reach_positions[r][c]) == 0
                    ).only_enforce_if(queens[r][c])

        # Maximize the number of queens placed
        model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        else:
            return []