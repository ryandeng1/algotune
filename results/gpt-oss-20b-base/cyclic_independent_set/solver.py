import itertools
import numpy as np
from numba import njit, int32, float64

@njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    N = children.shape[0]
    n = children.shape[1]
    result = np.empty(N, dtype=int32)
    res_len = 0
    neg_inf = -1e300
    blocked = np.full(N, False, dtype=bool)
    while True:
        # find best unblocked index
        best_idx = -1
        best_score = neg_inf
        for i in range(N):
            if not blocked[i] and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx == -1:
            break
        result[res_len] = best_idx
        res_len += 1
        blocked[best_idx] = True
        candidate = children[best_idx]
        for j in range(to_block.shape[0]):
            blocked_index = 0
            for k in range(n):
                blocked_index += ((candidate[k] + to_block[j, k]) % num_nodes) * powers[k]
            blocked[blocked_index] = True
    return result[:res_len]

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        # Precompute all candidate vertices
        children = np.array(
            list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32
        )
        # Compute initial scores (Python loop; can be accelerated if needed)
        scores = np.array(
            [self._priority(tuple(child), num_nodes, n) for child in children],
            dtype=float64,
        )
        # All possible shifts used for blocking
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32
        )
        # Precompute powers for index conversion
        powers = (num_nodes ** np.arange(n - 1, -1, -1)).astype(int32)

        # Call the accelerated numba solver
        selected_indices = solve_independent_set_numba(
            children, scores, to_block, powers, num_nodes
        )
        # Return the selected candidates as a list of tuples
        return [tuple(children[i]) for i in selected_indices]