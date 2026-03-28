from typing import Any
import itertools
import numba
import numpy as np
from numba.typed import List


@numba.njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    n = children.shape[1]
    N = children.shape[0]
    result = List()
    while True:
        best_idx = -1
        best_score = -np.inf
        for i in range(N):
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx == -1 or best_score == -np.inf:
            break
        result.append(best_idx)
        candidate = children[best_idx]
        for j in range(to_block.shape[0]):
            blocked_index = 0
            for k in range(n):
                blocked_index += (candidate[k] + to_block[j, k]) % num_nodes * powers[k]
            scores[blocked_index] = -np.inf
    return result

class Solver:

    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        """
        Solve the cyclic graph independent set problem.

        The task is to compute an optimal independent set in the n‑th strong product
        of a cyclic graph with num_nodes nodes. The solver uses a greedy algorithm that:
          1. Enumerates all candidate vertices (as n‑tuples).
          2. Computes a priority score for each candidate using the discovered priority function.
          3. Iteratively selects the candidate with the highest score and "blocks" conflicting nodes.

        This approach has been verified to match known optimal constructions.

        Args:
          problem (tuple): A tuple (num_nodes, n) representing the problem instance.

        Returns:
          List: A list of n-tuples representing the vertices in the independent set.
        """
        num_nodes, n = problem
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        scores = np.array([self._priority(tuple(child), num_nodes, n) for child in children])
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        powers = num_nodes ** np.arange(n - 1, -1, -1)
        selected_indices = solve_independent_set_numba(children, scores, to_block, powers, num_nodes)
        return [tuple(children[i]) for i in selected_indices]

    def _priority(self, el, num_nodes, n):
        """
        Compute the priority for a candidate node (represented as an n-tuple) in the
        independent set construction.

        This function clips the candidate values to ensure they do not exceed (num_nodes - 3)
        and then computes a score based on a weighted sum and modular arithmetic.
        Higher scores indicate a higher priority for inclusion.

        Args:
          el (tuple): An n-tuple candidate.
          num_nodes (int): Number of nodes in the base cyclic graph.
          n (int): Exponent (power) of the strong product.

        Returns:
          float: The computed priority score.
        """
        el_clipped = np.clip(el, a_min=None, a_max=num_nodes - 3)
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)))
        multipliers = np.array([num_nodes ** i for i in range(n - 1, -1, -1)], dtype=np.int32)
        x = np.sum((1 + values + el_clipped) * multipliers, axis=-1)
        return np.sum(x % (num_nodes - 2), dtype=float)
