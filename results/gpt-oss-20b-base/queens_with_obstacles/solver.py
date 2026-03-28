from typing import List, Tuple
import numpy as np

"""
The original solution built a large CP‑SAT model with 64 variables for a
single 8×8 board.  Constructing the model and solving it with OR‑Tools
is considerably slower than a simple back‑tracking search, especially for
small to medium boards.  The new implementation converts the board into
bitboard representations and performs a depth‑first search with
branch‑and‑bound.  The code remains a drop‑in replacement for the
original ``Solver.solve`` method.
"""

# --------------------------------------------------------------------------- #
# Utility functions for bitboard representation
# --------------------------------------------------------------------------- #

def pos_to_bit(row: int, col: int, n: int, m: int) -> int:
    """Return a bit with a 1 at the given cell in a flattened board."""
    return 1 << (row * m + col)


def build_attack_mask(n: int, m: int, obstacles: np.ndarray) -> List[int]:
    """
    For every cell (including obstacle cells) return a bitmask of
    all cells it attacks (including itself).  Obstacle cells are
    ignored during the recursion but need a mask for constant
    indexing.
    """
    masks = [0] * (n * m)
    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
    for r in range(n):
        for c in range(m):
            if obstacles[r, c]:
                continue
            mask = pos_to_bit(r, c, n, m)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while 0 <= nr < n and 0 <= nc < m and not obstacles[nr, nc]:
                    mask |= pos_to_bit(nr, nc, n, m)
                    nr += dr
                    nc += dc
            masks[r * m + c] = mask
    return masks


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #

class Solver:
    def solve(self, problem: np.ndarray) -> List[Tuple[int, int]]:
        """
        Finds a maximum set of non‑attacking queens on the given board.

        Parameters
        ----------
        problem : np.ndarray
            Binary matrix where `1` denotes an obstacle.

        Returns
        -------
        List[Tuple[int, int]]
            List of (row, col) positions of the placed queens.
        """
        n, m = problem.shape
        size = n * m
        obstacles_bits = 0
        for r in range(n):
            for c in range(m):
                if problem[r, c]:
                    obstacles_bits |= pos_to_bit(r, c, n, m)

        attack_masks = build_attack_mask(n, m, problem)

        all_cells = [i for i in range(size) if not (obstacles_bits >> i) & 1]
        best_solution: List[int] = []
        best_count = 0

        def dfs(placed_mask: int, candidates: List[int], depth: int):
            nonlocal best_solution, best_count

            # Upper bound: all remaining candidates could be taken
            if depth + len(candidates) <= best_count:
                return

            # Try to place a queen on each candidate in turn
            while candidates:
                idx = candidates.pop()
                bit = 1 << idx
                if placed_mask & bit:
                    continue  # already occupied by obstacle

                if placed_mask & attack_masks[idx]:
                    continue  # under attack

                new_mask = placed_mask | attack_masks[idx]

                # New candidates: remove all cells attacked by the new queen
                new_list = [c for c in candidates if not (new_mask & (1 << c))]

                dfs(new_mask, new_list, depth + 1)

            # Leaf: update best solution
            if depth > best_count:
                best_count = depth
                best_solution = [
                    (i // m, i % m) for i in range(size)
                    if (placed_mask >> i) & 1 and not (obstacles_bits >> i) & 1
                ]

        dfs(0, all_cells, 0)
        return best_solution