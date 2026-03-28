from typing import Any
from collections.abc import Iterator
import numpy as np


def queen_reach(instance: np.ndarray, start: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """
    Yields all coordinates that would be in reach of the queen, including the own position.

    Parameters:
        instance (np.ndarray): The chessboard matrix with obstacles.
        start (tuple): The starting position (row, column) of the queen.

    Yields:
        tuple: Coordinates (row, column) that the queen can reach.
    """
    n, m = instance.shape
    r, c = start
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in directions:
        nr, nc = (r + dr, c + dc)
        while 0 <= nr < n and 0 <= nc < m:
            if instance[nr, nc]:
                break
            yield (nr, nc)
            nr += dr
            nc += dc

class Solver:

    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        """
        Solves the Queens with Obstacles Problem using CP-SAT.

        Parameters:
            problem (np.ndarray): The chessboard matrix with obstacles.

        Returns:
            list: A list of tuples representing the positions (row, column) of the placed queens.
        """
        from ortools.sat.python import cp_model
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()
        queens = [[model.NewBoolVar(f'queen_{r}_{c}') for c in range(m)] for r in range(n)]
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)
                else:
                    pass
            else:
                pass
        else:
            pass
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_positions = list(queen_reach(instance, (r, c)))
                    print(f'Queen at ({r}, {c}) can reach: {reach_positions}')
                    model.Add(sum((queens[nr][nc] for nr, nc in reach_positions)) == 0).only_enforce_if(queens[r][c])
                else:
                    pass
            else:
                pass
        else:
            pass
        model.Maximize(sum((queens[r][c] for r in range(n) for c in range(m))))
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        else:
            return []
