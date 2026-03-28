from typing import List, Tuple
import sys

def solve(problem: 'np.ndarray') -> List[Tuple[int, int]]:
    """
    Fast backtracking solution for the Queens with Obstacles problem.

    Uses bitmask representation of rows, columns and both diagonals.  
    The algorithm places queens row‑by‑row and applies branch‑and‑bound
    pruning when the theoretical maximum achievable from the current
    state is less than the best solution already found.

    Parameters
    ----------
    problem : numpy.ndarray
        Binary board matrix where 1 indicates an obstacle, 0 an empty cell.

    Returns
    -------
    List[Tuple[int, int]]
        List of queen coordinates that form a maximum independent set.
    """
    import numpy as np

    n, m = problem.shape
    # Pre‑compute list of free cells per row
    free_rows = [list(np.where(problem[r] == 0)[0]) for r in range(n)]

    # Bit masks for diagonals.  diag1: r+c,  diag2: r-m+c
    diag1_size = n + m - 1
    diag2_size = n + m - 1

    best_board = []
    best_count = 0

    # Kinds of recursion: process rows one by one
    def dfs(r: int, cur_board: List[Tuple[int, int]],
            cols_mask: int,
            d1_mask: int,
            d2_mask: int,
            remaining_cells: int):
        nonlocal best_count, best_board

        # Upper bound: current count + remaining cells in rows r..n-1
        if len(cur_board) + remaining_cells <= best_count:
            return

        if r == n:
            if len(cur_board) > best_count:
                best_count = len(cur_board)
                best_board = cur_board.copy()
            return

        # Sort cells in this row by minimal conflicts if needed
        for c in free_rows[r]:
            bit_c = 1 << c
            d1 = r + c
            d2 = r - c + m - 1
            bit_d1 = 1 << d1
            bit_d2 = 1 << d2
            if (cols_mask & bit_c) or (d1_mask & bit_d1) or (d2_mask & bit_d2):
                continue
            # Place queen
            cur_board.append((r, c))
            dfs(r + 1, cur_board,
                cols_mask | bit_c,
                d1_mask | bit_d1,
                d2_mask | bit_d2,
                remaining_cells - 1)
            cur_board.pop()

        # Option: skip placing a queen in this row
        dfs(r + 1, cur_board, cols_mask, d1_mask, d2_mask, remaining_cells - 1)

    total_free = int(problem.sum() == 0)  # fallback if problem is scalar
    total_free = int(problem.sum() == 0)

    # Count total free cells
    total_free = int(problem.sum() == 0)
    total_free = int(problem.sum() == 0)

    # Compute total free cells
    total_free = int(problem.sum() == 0)
    total_free = problem.sum()

    dfs(0, [], 0, 0, 0, int(total_free))
    return best_board