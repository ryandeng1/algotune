# solver.py
from collections.abc import Iterator
import numpy as np
from ortools.sat.python import cp_model


def queen_reach(instance: np.ndarray, start: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """Yield all positions reachable by a queen from `start`, excluding obstacles."""
    n, m = instance.shape
    r, c = start
    for dr, dc in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < m:
            if instance[nr, nc]:
                break
            yield nr, nc
            nr += dr
            nc += dc


class Solver:
    """
    Efficient CP‑SAT solver for the Queens‑with‑Obstacles problem.
    Instead of enumerating all cells a queen can attack, we model the
    standard "one queen per row, column and diagonal" constraints.
    This reduces the number of constraints dramatically for large boards.
    Obstacles are handled by forcing the corresponding cell variable to
    be 0.
    """

    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        """
        Return the coordinates of a maximum‑cardinality placement of queens.
        The board `problem` is a 2‑D numpy array with 0 = empty, 1 = obstacle.
        """
        instance = problem
        n, m = instance.shape

        # ---- model ---------------------------------------------------------
        model = cp_model.CpModel()

        # Flattened variables for faster indexing
        queen_vars = np.zeros((n, m), dtype=object)
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    queen_vars[r, c] = 0  # obstacle
                else:
                    queen_vars[r, c] = model.NewBoolVar(f"q{r}_{c}")

        # Row constraints: at most one queen per row
        for r in range(n):
            model.AddAtMostOne([queen_vars[r, c] for c in range(m) if queen_vars[r, c] is not 0])

        # Column constraints: at most one queen per column
        for c in range(m):
            model.AddAtMostOne([queen_vars[r, c] for r in range(n) if queen_vars[r, c] is not 0])

        # Diagonal constraints (↗↘)
        for d in range(-(n - 1), m):
            diag = [queen_vars[r, c] for r, c in enumerate(range(max(0, -d), min(n, m - d))) if queen_vars[r, c] is not 0]
            if diag:
                model.AddAtMostOne(diag)

        # Diagonal constraints (↘↙)
        for d in range(n + m - 1):
            diag = []
            for r in range(n):
                c = d - r
                if 0 <= c < m and queen_vars[r, c] is not 0:
                    diag.append(queen_vars[r, c])
            if diag:
                model.AddAtMostOne(diag)

        # Maximise total number of queens
        model.Maximize(sum(v for row in queen_vars for v in row if v is not 0))

        # ---- solve ---------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for r in range(n) for c in range(m)
                    if queen_vars[r, c] and solver.Value(queen_vars[r, c]) == 1]
        return []