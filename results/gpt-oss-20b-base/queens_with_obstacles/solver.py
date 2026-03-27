import numpy as np
from typing import Iterator, Tuple, List


def queen_reach(instance: np.ndarray, start: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
    n, m = instance.shape
    r, c = start
    # Order of directions: 8 around the queen
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < m:
            if instance[nr, nc]:
                break
            yield nr, nc
            nr += dr
            nc += dc


class Solver:
    def solve(self, problem: np.ndarray) -> List[Tuple[int, int]]:
        from ortools.sat.python import cp_model

        n, m = problem.shape
        model = cp_model.CpModel()

        # Decision variables
        queens = [[model.NewBoolVar(f"q_{r}_{c}") for c in range(m)] for r in range(n)]

        # No queen on an obstacle
        for r in range(n):
            for c in range(m):
                if problem[r, c]:
                    model.Add(queens[r][c] == 0)

        # Pre‑compute reachability for every free cell
        reach = {}
        for r in range(n):
            for c in range(m):
                if not problem[r, c]:
                    reach[(r, c)] = list(queen_reach(problem, (r, c)))

        # Ensure no two queens attack each other
        for (r, c), cells in reach.items():
            if cells:
                # Sum over all reachable cells must be 0 if a queen is placed here
                sum_vars = [queens[nr][nc] for nr, nc in cells]
                model.Add(sum(sum_vars) == 0).OnlyEnforceIf(queens[r][c])

        # Maximise total number of queens
        model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        return []