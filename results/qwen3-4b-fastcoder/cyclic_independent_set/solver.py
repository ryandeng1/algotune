from typing import Any
import itertools
import numpy as np
from numba import njit

@njit
def _solve_independent_set(children, scores, to_block, powers, num_nodes):
    n = children.shape[1]
    N = children.shape[0]
    selected = np.zeros(N, dtype=np.int32)
    count = 0
    while True:
        best_idx = -1
        best_score = -10**18
        for i in range(N):
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx == -1:
            break
        selected[count] = best_idx
        count += 1
        candidate = children[best_idx]
        for j in range(to_block.shape[0]):
            blocked_index = 0
            for k in range(n):
                shifted = (candidate[k] + to_block[j, k]) % num_nodes
                blocked_index += shifted * powers[k]
            scores[blocked_index] = -10**18
    return selected[:count]

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        scores = np.array([self._priority(tuple(child), num_nodes, n) for child in children])
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        powers = num_nodes ** np.arange(n - 1, -1, -1)
        selected_indices = _solve_independent_set(children, scores, to_block, powers, num_nodes)
        return [tuple(children[i]) for i in selected_indices]