import numpy as np
import itertools
from numba import njit
from numba.typed import List

@njit
def _select_set(children, scores, to_block, powers, num_nodes):
    N = children.shape[0]
    res = List()
    while True:
        best = -1
        best_score = -np.inf
        for i in range(N):
            s = scores[i]
            if s > best_score:
                best_score = s
                best = i
        if best == -1 or best_score == -np.inf:
            break
        res.append(best)
        cand = children[best]
        for j in range(to_block.shape[0]):
            idx = 0
            for k in range(powers.shape[0]):
                idx += (cand[k] + to_block[j, k]) % num_nodes * powers[k]
            scores[idx] = -np.inf
    return res

def _priority_array(children, num_nodes, n):
    # vectorised priority computation
    # values = 2 * product(range(1,n), repeat=n)
    vals = np.multiply.reduce(
        [np.arange(1, n) for _ in range(children.shape[1])],
        axis=None
    )
    # but easiest to compute directly using broadcasting
    mult = np.array([num_nodes**i for i in range(n-1, -1, -1)], dtype=np.int64)
    el = children.astype(np.int64)
    values = 2 * np.sum(
        [(np.arange(1, n)[:, None, None, None] if i==0 else 1) for i in range(n)],
        axis=0
    )
    # speed: actually compute numeric array
    # compute sum over i
    x = 0
    for i in range(n):
        x += (1 + 2 * np.arange(1, n)[None] + el[:, i:i+1]) * mult[i]
    return np.sum(np.mod(x, num_nodes-2).astype(np.float64), axis=1)

class Solver:
    def solve(self, problem):
        num_nodes, n = problem
        # precompute children
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        scores = _priority_array(children, num_nodes, n)
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        powers = np.array([num_nodes**i for i in range(n-1, -1, -1)], dtype=np.int64)
        ind = _select_set(children, scores, to_block, powers, num_nodes)
        return [tuple(children[i]) for i in ind]