from __future__ import annotations
from typing import List, Tuple
import numpy as np
from ortools.sat.python import cp_model

def _precompute_attacks(board: np.ndarray) -> List[List[int]]:
    """Return, for each cell, the linear indices of all cells it attacks."""
    n, m = board.shape
    size = n * m
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    attacks: List[List[int]] = [[] for _ in range(size)]
    empty = board.ravel() == 0
    for r in range(n):
        for c in range(m):
            if board[r, c]:
                continue
            idx0 = r * m + c
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while 0 <= nr < n and 0 <= nc < m:
                    if board[nr, nc]:
                        break
                    attacks[idx0].append(nr * m + nc)
                    nr += dr
                    nc += dc
    return attacks, empty

class Solver:
    def solve(self, problem: np.ndarray) -> List[Tuple[int, int]]:
        n, m = problem.shape
        attacks, empty = _precompute_attacks(problem)
        flat_size = n * m

        model = cp_model.CpModel()
        queens = [model.NewBoolVar(f'q_{i}') for i in range(flat_size)]

        # Enforce obstacles
        for i, is_empty in enumerate(empty):
            if not is_empty:
                model.Add(queens[i] == 0)

        # Mutual attack constraints
        for idx in range(flat_size):
            if not empty[idx]:
                continue
            for atk in attacks[idx]:
                # Use a single linear constraint per pair
                model.Add(queens[idx] + queens[atk] <= 1)

        # Objective: maximize number of queens
        model.Maximize(sum(queens))

        solver = cp_model.CpSolver()
        # Optional: limit solving time if you want to avoid long runtimes
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result = []
            for idx in range(flat_size):
                if solver.Value(queens[idx]):
                    result.append((idx // m, idx % m))
            return result
        return []