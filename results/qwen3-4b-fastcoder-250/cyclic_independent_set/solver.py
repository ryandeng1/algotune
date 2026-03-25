import numpy as np
from numba import njit
import itertools

@njit
def _priority(el, num_nodes, n):
    num_nodes = 7
    el_clipped = np.clip(el, a_min=None, a_max=num_nodes - 3)
    multipliers = np.zeros(n, dtype=np.int32)
    for i in range(n):
        multipliers[i] = num_nodes ** (n - 1 - i)
    values = np.zeros(n, dtype=np.int32)
    for i in range(n):
        values[i] = 2 * (i + 1)
    x = 0
    for i in range(n):
        x += (1 + values[i] + el_clipped[i]) * multipliers[i]
    return x % (num_nodes - 2)

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        scores = np.zeros(children.shape[0], dtype=np.float64)
        for i in range(children.shape[0]):
            scores[i] = _priority(children[i], num_nodes, n)
        top_indices = np.argsort(scores)[::-1]
        return [tuple(children[i]) for i in top_indices[:10]]
