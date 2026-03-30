# solver.py
from typing import Any
import itertools
import numpy as np


def _compute_scores(children: np.ndarray, num_nodes: int, n: int) -> np.ndarray:
    """
    Vectorised version of the original priority computation.
    """
    # clip children values
    el_clipped = np.clip(children, None, num_nodes - 3)

    # pre‑compute the constant part that does not depend on the child
    values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int32)
    multipliers = np.array([num_nodes ** i for i in range(n - 1, -1, -1)], dtype=np.int32)

    # broadcast the operations
    x = (1 + values + el_clipped) * multipliers
    # Sum along columns to get the weighted sum, then take modulo
    scores = np.sum(x, axis=1) % (num_nodes - 2)
    return scores.astype(np.float64)


def _solve_independent_set(
    children: np.ndarray,
    scores: np.ndarray,
    to_block: np.ndarray,
    powers: np.ndarray,
    num_nodes: int,
) -> list[int]:
    """
    Fast non‑numba implementation of the greedy independent set algorithm.
    Uses NumPy vectorisation for the blocking step and ``argmax`` for the
    selection step.
    """
    N, n = children.shape
    result: list[int] = []

    while True:
        # Find the best available candidate
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        if best_score == -np.inf:
            break

        result.append(best_idx)

        # Block all conflicting vertices
        candidate = children[best_idx]                    # shape (n,)
        blocked = (candidate + to_block) % num_nodes      # shape (m, n)
        blocked_indices = blocked.dot(powers)            # shape (m,)
        scores[blocked_indices] = -np.inf

    return result


class Solver:
    """
    Solver for the cyclic graph independent set problem.

    The implementation is heavily optimised by avoiding loops in Python, using
    NumPy vectorisation for all expensive operations, and a custom scoring
    routine that fits the description of the original problem.
    """

    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem

        # 1. Enumerate all candidate vertices (n‑tuples)
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)

        # 2. Compute priority scores in a vectorised manner
        scores = _compute_scores(children, num_nodes, n)

        # 3. Pre‑compute the blocking offsets and powers for linear indexing
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)),
            dtype=np.int32,
        )
        powers = (num_nodes ** np.arange(n - 1, -1, -1)).astype(np.int64)

        # 4. Run the greedy algorithm
        selected_indices = _solve_independent_set(
            children, scores, to_block, powers, num_nodes
        )

        # 5. Convert indices back to tuples
        return [tuple(children[i]) for i in selected_indices]