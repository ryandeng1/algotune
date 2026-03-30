# solver.py

# The goal of this solver is to maximise the number of queens that can be placed
# on a board with obstacles (``True`` in the input array means an obstacle).
#
# The CP‑SAT model below is intentionally compact: each cell is represented by
# a Boolean variable and the model only contains the necessary constraints.
# Printing during model building is removed to keep the construction time tiny.
#
# The search strategy is left to CP‑SAT's default – on small to medium boards
# it typically finds a good solution quickly.  For very large boards a
# custom search strategy could be added later if needed.

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from ortools.sat.python import cp_model


# --------------------------------------------------------------------------- #
# Helper: compute the reach of a queen at a given cell.
# --------------------------------------------------------------------------- #
def _queen_reach(
    board: np.ndarray,
    position: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Return the list of cells that a queen placed at ``position`` can attack
    (excluding obstacles, excluding its own cell).
    """
    n, m = board.shape
    r, c = position
    reaches: List[Tuple[int, int]] = []

    # Eight knight‑like directions (queen moves)
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
            if board[nr, nc]:  # obstacle
                break
            reaches.append((nr, nc))
            nr += dr
            nc += dc
    return reaches


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class Solver:
    """
    Solve the *Queens with Obstacles* problem using OR‑Tools CP‑SAT.
    """

    def __init__(self) -> None:
        # We expect the model to be rebuilt for every call to ``solve`` –
        # all internal data is therefore created lazily.
        pass

    # ------------------------------- #
    # Main API (called by the judge)
    # ------------------------------- #
    def solve(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """
        Place as many queens as possible on ``board`` without two queens
        attacking each other and without one being on an obstacle.

        Parameters
        ----------
        board : np.ndarray
            A 2‑D NumPy array of booleans where ``True`` indicates an
            obstacle (cannot host a queen).

        Returns
        -------
        List[Tuple[int, int]]
            One (row, col) pair for each queen in the optimal arrangement.
        """
        n, m = board.shape
        model = cp_model.CpModel()

        # ---------- Variables ----------
        # Use a flat list for easier handling.
        queens = np.empty((n, m), dtype=object)
        for r in range(n):
            for c in range(m):
                queens[r, c] = model.NewBoolVar(f"q_{r}_{c}")

        # ---------- Obstacle constraints ----------
        for r in range(n):
            for c in range(m):
                if board[r, c]:
                    model.Add(queens[r, c] == 0)

        # ---------- Attack constraints ----------
        # For each free cell we pre‑compute the attacked cells
        # and ensure no other queen sits there if we use this cell.
        for r in range(n):
            for c in range(m):
                if board[r, c]:
                    continue
                reachable = _queen_reach(board, (r, c))
                if reachable:  # only add a guard if something to guard
                    # sum(queens[reachable]) == 0  if queens[r,c] == 1
                    # Equivalent to: queens[r,c]*sum(queens[reachable]) == 0
                    # This can be written as:
                    #   sum(queens[reachable]) <= (1 - queens[r,c]) * INF
                    # But CP‑SAT understands the direct formulation below.
                    model.Add(
                        sum(queens[nr, nc] for nr, nc in reachable) == 0
                    ).OnlyEnforceIf(queens[r, c])

        # ---------- Objective ----------
        model.Maximize(
            sum(queens[r, c] for r in range(n) for c in range(m))
        )

        # ---------- Solve ----------
        solver = cp_model.CpSolver()
        # We do not want the solver to flood the output; we can disable it.
        solver.parameters.log_search_progress = False
        # A small, generous time budget – CP‑SAT can prune a lot on its own.
        solver.parameters.max_time_in_seconds = 10.0

        status = solver.Solve(model)

        if status in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):
            result = [
                (r, c) for r in range(n) for c in range(m) if solver.Value(queens[r, c])
            ]
            return result
        else:
            return []