import numpy as np
from ortools.sat.python import cp_model

# Pre‑compute all queen attack masks for a board of a given size.
# Each cell is represented by a tuple (row, col).
def _build_attack_dict(n: int, m: int, obstacles: np.ndarray):
    attacks = {}
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]
    for r in range(n):
        for c in range(m):
            if obstacles[r, c]:
                continue
            cells = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < n and 0 <= nc < m and not obstacles[nr, nc]:
                    cells.append((nr, nc))
                    nr += dr
                    nc += dc
            attacks[(r, c)] = cells
    return attacks


class Solver:
    """
    Fast solver for the Queens with Obstacles Problem.
    Uses cp_sat only once with pre‑computed attack lists.
    """

    def __init__(self):
        # nothing to initialise
        pass

    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        """Return a maximal set of non‑attacking queens on an obstacle board."""
        n, m = problem.shape
        attacks = _build_attack_dict(n, m, problem)

        # CP‑SAT model
        model = cp_model.CpModel()
        # Boolean var for every non‑obstacle cell
        vars_ = {cell: model.NewBoolVar(f"q{cell[0]}_{cell[1]}") for cell in attacks}

        # Constraints: if a queen is placed then it cannot attack other queens
        for cell, reach in attacks.items():
            if reach:  # only add if there are reachable cells
                model.Add(sum(vars_[q] for q in reach) == 0).OnlyEnforceIf(vars_[cell])

        # Objective: maximise number of queens
        model.Maximize(sum(vars_.values()))

        # Solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # guard against endless runtimes
        solver.parameters.num_search_workers = 8  # use parallel workers
        status = solver.Solve(model)

        # Extract solution
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for (r, c), b in vars_.items() if solver.Value(b)]
        return []