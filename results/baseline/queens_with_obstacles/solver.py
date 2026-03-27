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
    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),  # Up-left, Up, Up-right
        (0, -1),
        (0, 1),  # Left, Right
        (1, -1),
        (1, 0),
        (1, 1),  # Down-left, Down, Down-right
    ]

    # yield (r, c)  # Own position

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < m:
            if instance[nr, nc]:  # Stop if there's an obstacle
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

        # Decision variables
        queens = [[model.NewBoolVar(f"queen_{r}_{c}") for c in range(m)] for r in range(n)]

        # Constraint: No queens on obstacles
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)

        # Constraint: No two queens attack each other
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_positions = list(queen_reach(instance, (r, c)))
                    print(f"Queen at ({r}, {c}) can reach: {reach_positions}")
                    # If we place a queen at (r, c), ensure no other queens are in reach
                    model.Add(
                        sum(queens[nr][nc] for nr, nc in reach_positions) == 0
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
