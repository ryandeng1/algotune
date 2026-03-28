import numpy as np
from ortools.sat.python import cp_model
from typing import List, Tuple


def _queen_reach(instance: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
    """Return all empty cells reachable from (r, c) in all 8 directions."""
    n, m = instance.shape
    cells = []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1),  (1, 0), (1, 1)):
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < m:
            if instance[nr, nc]:
                break
            cells.append((nr, nc))
            nr += dr
            nc += dc
    return cells


class Solver:
    def solve(self, problem: np.ndarray) -> List[Tuple[int, int]]:
        """
        Solve the Queens with Obstacles Problem. The board is represented as a 2-D numpy
        boolean array where `True` indicates an obstacle.  The function returns a list of
        coordinates where queens can be placed such that no queen can attack another.
        """
        # Pre‑filter empty cells and compute reachable sets
        n, m = problem.shape
        empty_cells = [(r, c) for r in range(n) for c in range(m) if not problem[r, c]]
        reach_map = {cell: _queen_reach(problem, *cell) for cell in empty_cells}

        model = cp_model.CpModel()
        # Create a variable for each empty cell
        var = {(r, c): model.NewBoolVar(f'q_{r}_{c}') for r, c in empty_cells}

        # For every queen variable, ensure that no other queen lies in its reach area
        for (r, c), reach in reach_map.items():
            if reach:
                model.Add(sum(var[pos] for pos in reach) == 0).OnlyEnforceIf(var[(r, c)])

        # Objective: maximize number of placed queens
        model.Maximize(sum(var.values()))

        solver = cp_model.CpSolver()
        # Small speed improvements
        solver.parameters.search_branching = cp_model.BREADTH_FIRST_SEARCH
        solver.parameters.rational_processing = True

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for (r, c), v in var.items() if solver.Value(v)]
        return []