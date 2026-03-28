import numpy as np
from functools import lru_cache
from typing import List, Tuple

def _bit_positions(bits: int, n: int, m: int) -> List[Tuple[int, int]]:
    """Return list of (row, col) for set bits in flattened board."""
    pos = []
    idx = 0
    while bits:
        t = bits & -bits
        r = idx + (t.bit_length() - 1)
        pos.append((r // m, r % m))
        idx += t.bit_length()
        bits ^= t
    return pos

def solve(problem: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solve the Queens with Obstacles Problem using a fast recursive search
    with bit masks. The board is flattened into a single integer bit mask
    where a 1-bit represents an *available* square (not an obstacle).
    """
    n, m = problem.shape
    all_mask = 0
    for r in range(n):
        for c in range(m):
            if not problem[r, c]:
                all_mask |= 1 << (r * m + c)

    # Pre‑compute attack masks for every position
    attack = {}
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    for r in range(n):
        for c in range(m):
            if problem[r, c]:
                continue
            mask = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < n and 0 <= nc < m:
                    if problem[nr, nc]:
                        break
                    mask |= 1 << (nr * m + nc)
                    nr += dr
                    nc += dc
            attack[r * m + c] = mask

    @lru_cache(maxsize=None)
    def dfs(available: int) -> Tuple[int, int]:
        """Return (best_count, best_mask) for the given available squares."""
        if available == 0:
            return 0, 0

        # Choose first available square as a pivot
        pivot = (available & -available).bit_length() - 1
        best_cnt, best = 0, 0

        # Branch 1: put queen at pivot
        new_avail = available & ~((1 << pivot) | attack[pivot])
        cnt1, mask1 = dfs(new_avail)
        cnt1 += 1
        mask1 |= 1 << pivot
        if cnt1 > best_cnt:
            best_cnt, best = cnt1, mask1

        # Branch 2: skip pivot
        new_avail = available & ~(1 << pivot)
        cnt2, mask2 = dfs(new_avail)
        if cnt2 > best_cnt:
            best_cnt, best = cnt2, mask2

        return best_cnt, best

    _, bestmask = dfs(all_mask)
    return _bit_positions(bestmask, n, m)