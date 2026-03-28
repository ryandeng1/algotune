import itertools
import numpy as np
from numba import njit, types
from numba.typed import List

@njit
def _solve(children, scores, to_block, powers, num_nodes):
    N, n = children.shape
    blocked = np.full(N, False, dtype=nb.boolean)
    res = List()
    while True:
        # find highest score among unblocked candidates
        best_idx = -1
        best_score = -1e308
        for i in range(N):
            if blocked[i]:
                continue
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx == -1:
            break
        res.append(best_idx)
        candidate = children[best_idx]
        # block all vertices that conflict with candidate
        for j in range(to_block.shape[0]):
            blocked_idx = 0
            for k in range(n):
                blocked_idx += ((candidate[k] + to_block[j, k]) % num_nodes) * powers[k]
            scores[blocked_idx] = -1e308  # mark as blocked
            blocked[blocked_idx] = True
    return res

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem

        # Pre‑compute all tuples of the graph
        children = np.array(
            list(itertools.product(range(num_nodes), repeat=n)),
            dtype=np.int32,
        )
        total = children.shape[0]

        # Vectorised priority computation
        # Clip values
        clipped = np.clip(children, 0, num_nodes - 3).astype(np.int64)

        # Powers for index calculation
        pow_n = np.array([num_nodes ** i for i in range(n - 1, -1, -1)], dtype=np.int64)

        # base part of the score that is independent of the clip
        # we compute it once using all combinations of (1 + 1..n-1)
        comb = np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int64)
        base = np.sum((1 + comb) * pow_n, axis=1)

        # Scores: (base + clipped.dot(pow_n)) % (num_nodes-2), summed
        offset = clipped.dot(pow_n)
        scores = np.sum((base + offset) % (num_nodes - 2), axis=0)

        # All possible offsets (-1,0,1) for blocking
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)),
            dtype=np.int32,
        )

        selected = _solve(children, scores, to_block, pow_n, num_nodes)
        return [tuple(children[i]) for i in selected]