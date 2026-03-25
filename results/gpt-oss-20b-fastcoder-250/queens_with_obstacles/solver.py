# solver.py

import numpy as np
from ortools.sat.python import cp_model
from collections.abc import Iterator, Iterable

def queen_reach(instance: np.ndarray, start: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """Yield all coordinates reachable by a queen from `start`,
    stopping at obstacles or board edges. The queen's own position
    is *not* yielded."""
    n, m = instance.shape
    r, c = start
    # Eight directions: N, NE, E, SE, S, SW, W, NW
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0),  (1, -1), (0, -1), (-1, -1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < m:
            if instance[nr, nc]:      # obstacle blocks sight
                break
            yield (nr, nc)
            nr += dr
            nc += dc

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> list[tuple[int, int]]:
        """Use CP-SAT to maximise the number of non‑attacking queens
        on a board with obstacles."""
        instance = problem
        n, m = instance.shape

        # Build CP-SAT model
        model = cp_model.CpModel()

        # Decision variables: 1 if a queen is placed at (r,c)
        queens = [[model.NewBoolVar(f"q_{r}_{c}") for c in range(m)] for r in range(n)]

        # No queens on obstacles
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)

        # For each cell, if a queen is placed there, forbid queens in reach
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach = list(queen_reach(instance, (r, c)))
                    # If no other queen in reach, sum must be 0 when cell occupied
                    if reach:
                        # Build linear expression for queens in reach
                        reach_sum = sum(queens[rr][cc] for rr, cc in reach)
                        # Only enforce when we place a queen here
                        model.Add(reach_sum == 0).OnlyEnforceIf(queens[r][c])

        # Objective: maximise number of queens
        model.Maximize(
            sum(queens[r][c] for r in range(n) for c in range(m))
        )

        # Linear programming solver
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8  # use all cores for speed
        solver.parameters.max_time_in_seconds = kwargs.get("timeout", 10.0)
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for r in range(n) for c in range(m)
                    if solver.Value(queens[r][c])]
        return []

# End of solver.py
