from typing import Any
import itertools
import numpy as np
from numba import njit
from numba.typed import List


@njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    N = children.shape[0]
    n = children.shape[1]
    result = List()
    while True:
        # Find candidate with maximum score
        best_idx = -1
        best_score = -1e300
        for i in range(N):
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx == -1 or best_score == -1e300:
            break
        result.append(best_idx)
        candidate = children[best_idx]
        # Block all conflicting nodes
        for j in range(to_block.shape[0]):
            blocked_index = 0
            shift = to_block[j]
            for k in range(n):
                blocked_index += (((candidate[k] + shift[k]) % num_nodes) * powers[k])
            scores[blocked_index] = -1e300
    return result


class Solver:
    def _priority(self, tup, num_nodes, n):
        # Example placeholder priority function
        return sum(tup) * 0.01

    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        # Generate all candidates
        children = np.fromiter(itertools.product(range(num_nodes), repeat=n),
                               dtype=np.int32, count=num_nodes**n).reshape(num_nodes**n, n)
        # Compute scores
        scores = np.empty(children.shape[0], dtype=np.float64)
        for idx, child in enumerate(children):
            scores[idx] = self._priority(tuple(child), num_nodes, n)
        # Shifts for blocking
        to_block = np.fromiter(itertools.product([-1, 0, 1], repeat=n),
                               dtype=np.int32, count=3**n).reshape(3**n, n)
        # Powers for index conversion
        powers = np.array([num_nodes**i for i in reversed(range(n))], dtype=np.int32)

        selected_indices = solve_independent_set_numba(
            children, scores, to_block, powers, num_nodes
        )
        return [tuple(children[i]) for i in selected_indices]