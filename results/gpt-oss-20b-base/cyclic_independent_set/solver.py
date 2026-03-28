import itertools
import numpy as np
from numba import njit
from numba.typed import List

@njit
def _blocking(num_nodes, children, to_block, powers, scores):
    """Inner loop that selects vertices and blocks their neighbours."""
    N = children.shape[0]
    n = children.shape[1]
    result = List()
    while True:
        best_idx = -1
        best_score = -1e308
        # linear search for max – N is small enough
        for i in range(N):
            s = scores[i]
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx == -1 or best_score == -1e308:
            break
        result.append(best_idx)
        cand = children[best_idx]
        # block all neighbours (including itself)
        for j in range(to_block.shape[0]):
            # compute merged index: sum (cand[k]+to_block[j,k])%num_nodes * powers[k]
            idx = 0
            t = to_block[j]
            for k in range(n):
                idx += (cand[k] + t[k]) % num_nodes * powers[k]
            scores[idx] = -1e308
    return result

class Solver:
    def _priority_vectorized(self, children, num_nodes, n):
        """
        Vectorised computation of the priority score used in the greedy
        independent‑set construction.
        """
        # clip along axis 0
        cl = np.clip(children, None, num_nodes - 3)
        # create the values array: 2 * 1..(n-1) repeated per coordinate
        val = 2 * np.arange(1, n, dtype=np.int32)
        # expand to shape (N, n)
        vals = np.tile(val, (children.shape[0], 1))
        mult = num_nodes ** np.arange(n - 1, -1, -1, dtype=np.int32)
        x = (1 + vals + cl) * mult  # broadcasting -> (N, n)
        x = x.sum(axis=1) % (num_nodes - 2)
        # the original code summed all entries but returned a scalar;
        # we need a single score per vertex, so sum over the n coordinates
        # after mod which keeps the behaviour but is still per‑vertex.
        return x.astype(np.float64)

    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        # generate all candidate vertices
        children = np.array(
            list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32
        )
        # priority scores (one scalar per vertex)
        scores = self._priority_vectorized(children, num_nodes, n)
        # neighbours to block (±1, 0) per coordinate
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32
        )
        powers = num_nodes ** np.arange(n - 1, -1, -1, dtype=np.int32)
        selected = _blocking(num_nodes, children, to_block, powers, scores)
        return [tuple(children[i]) for i in selected]