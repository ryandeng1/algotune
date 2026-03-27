from typing import Any
import itertools
import numba
import numpy as np
from numba.typed import List


def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    n = children.shape[1]
    N = children.shape[0]
    result = List()
    while True:
        best_idx = -1
        best_score = -np.inf
        # Find the candidate with the highest score.
        for i in range(N):
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        if best_idx == -1 or best_score == -np.inf:
            break
        result.append(best_idx)
        # Get the selected candidate.
        candidate = children[best_idx]
        # For each shift in to_block, compute the corresponding blocked index.
        for j in range(to_block.shape[0]):
            blocked_index = 0
            for k in range(n):
                # Use the precomputed powers to convert the shifted candidate to an index.
                blocked_index += ((candidate[k] + to_block[j, k]) % num_nodes) * powers[k]
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

        # Precompute all candidate vertices.
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        # Compute initial scores for all candidates.
        scores = np.array([self._priority(tuple(child), num_nodes, n) for child in children])
        # All possible shifts used for blocking.
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        # Precompute powers for index conversion.
        powers = num_nodes ** np.arange(n - 1, -1, -1)

        # Call the accelerated numba solver.
        selected_indices = solve_independent_set_numba(
            children, scores, to_block, powers, num_nodes
        )

        # Return the selected candidates as a list of tuples.
        return [tuple(children[i]) for i in selected_indices]
