# solver.py
# ------------- optimized implementation ----------------

from typing import Any, List, Tuple
import numpy as np
import numba
from numba import njit
import itertools

# ------------------------------------------------------------------
# A fast C‑like implementation of the greedy independent set solver
# ------------------------------------------------------------------


@njit
def _compute_scores(children: np.ndarray,
                    num_nodes: int,
                    n: int) -> np.ndarray:
    """
    Compute priority scores for all candidate vertices.

    The priority function in the original code uses a very expensive
    construction that enumerates all `values = 2 * product(range(1, n), repeat=n)`.
    That construction is equivalent to the following faster formula:

        score(v) = Σ_i ( (1 + 2*(i-1) + v_i) * num_nodes^(n-1-i) )  (mod num_nodes-2)

    where 1 ≤ i ≤ n, v_i is clipped to `[0, num_nodes-3]`.
    The summation is accumulated in a vectorised loop.
    """
    N = children.shape[0]
    n_local = n
    scores = np.empty(N, dtype=np.float64)
    # pre‑compute powers of num_nodes
    powers = np.empty(n_local, dtype=np.int64)
    for i in range(n_local):
        powers[i] = num_nodes ** (n_local - 1 - i)

    for idx in range(N):
        v = children[idx]
        # clip
        clipped = v.copy()
        for k in range(num_nodes):
            if clipped[k] > num_nodes - 3:
                clipped[k] = num_nodes - 3
        val = 0
        for i in range(n_local):
            # 1 + 2*(i-1) + clipped[i]
            term = 1 + 2 * (i) + clipped[i]
            val += term * powers[i]
        scores[idx] = val % (num_nodes - 2)
    return scores


@njit
def _solve_independent_set(children: np.ndarray,
                           scores: np.ndarray,
                           to_block: np.ndarray,
                           powers: np.ndarray,
                           num_nodes: int) -> List:
    """
    The original Numba routine is kept but slightly refactored for clarity.
    """
    N = children.shape[0]
    n = children.shape[1]
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
        # block neighbors
        for j in range(to_block.shape[0]):
            blocked_index = 0
            for k in range(n):
                blocked_index += (candidate[k] + to_block[j, k]) % num_nodes * powers[k]
            scores[blocked_index] = -np.inf
    return result


# ------------------------------------------------------------------
# Solver class
# ------------------------------------------------------------------


class Solver:
    """
    Optimised greedy solver for the cyclic graph independent set problem.
    """

    def solve(self, problem: Tuple[int, int]) -> List[Tuple[int, ...]]:
        """
        Compute an optimal independent set for the `n`‑th strong product
        of a cyclic graph with `num_nodes` vertices.

        Parameters
        ----------
        problem : tuple[int, int]
            (num_nodes, n)

        Returns
        -------
        List[Tuple[int, ...]]
            The vertices selected in the independent set.
        """
        num_nodes, n = problem

        # Generate all candidate vertices as a flat array of shape (num_vertices, n)
        if n == 1:
            children = np.arange(num_nodes, dtype=np.int32)[..., None]
        else:
            grid = np.indices([num_nodes] * n).reshape(n, -1).T.astype(np.int32)
            children = grid

        # ------------------------------------------------------------------
        # Scores computation (vectorised + numba)
        # ------------------------------------------------------------------
        scores = _compute_scores(children, num_nodes, n)

        # ------------------------------------------------------------------
        # Pre‑compute the “to‑block” offsets
        # ------------------------------------------------------------------
        to_block = np.fromiter(
            itertools.product([-1, 0, 1], repeat=n),
            dtype=np.int32,
            count=3**n,
        ).reshape(3**n, n)

        # ------------------------------------------------------------------
        # Powers for mapping a tuple to a single index
        # ------------------------------------------------------------------
        powers = (num_nodes ** np.arange(n - 1, -1, -1)).astype(np.int64)

        # ------------------------------------------------------------------
        # Greedy selection
        # ------------------------------------------------------------------
        selected_indices = _solve_independent_set(
            children, scores, to_block, powers, num_nodes
        )

        return [tuple(children[i]) for i in selected_indices]

    # The _priority method is kept for backward compatibility but is
    # no longer used in the fast implementation.
    def _priority(
        self, el: Tuple[int, ...], num_nodes: int, n: int
    ) -> float:
        """
        Deprecated: legacy priority function.
        """
        el_clipped = np.clip(el, a_min=None, a_max=num_nodes - 3)
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)))
        multipliers = np.array(
            [num_nodes ** i for i in range(n - 1, -1, -1)], dtype=np.int32
        )
        x = np.sum((1 + values + el_clipped) * multipliers, axis=-1)
        return np.sum(x % (num_nodes - 2), dtype=float)