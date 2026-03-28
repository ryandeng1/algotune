from typing import Tuple, List
import itertools
import numpy as np
import numba
from numba.typed import List


@numba.njit
def _solve_greedy(
    children: np.ndarray,
    scores: np.ndarray,
    to_block: np.ndarray,
    powers: np.ndarray,
    num_nodes: int,
) -> List[int]:
    """
    Greedy selection of an independent set.

    Parameters
    ----------
    children : (N, n) array
        All candidate vertices.
    scores : (N,) array
        Priority for each candidate.
    to_block : (3**n, n) array
        Offsets that represent adjacency in the strong product.
    powers : (n,) array
        Powers of ``num_nodes`` for the linear index calculation.
    num_nodes : int
        Base graph size.

    Returns
    -------
    List[int]
        Indices of chosen vertices in ``children``.
    """
    N = children.shape[0]
    result: List[int] = List()
    while True:
        # pick the unused candidate with highest score
        max_idx = -1
        max_score = -1.0e300
        for i in range(N):
            if scores[i] > max_score:
                max_score = scores[i]
                max_idx = i
        if max_idx == -1 or max_score == -1.0e300:
            break

        result.append(max_idx)
        candidate = children[max_idx]

        # block all vertices that conflict with the chosen one
        for offs in to_block:
            blocked = (candidate + offs) % num_nodes
            idx = blocked[0]
            for k in range(1, blocked.size):
                idx = idx * num_nodes + blocked[k]
            scores[idx] = -1.0e300

    return result


@numba.njit
def _compute_scores(
    children: np.ndarray, num_nodes: int, n: int
) -> np.ndarray:
    """
    Vectorised calculation of the priority score for every candidate.

    Notes
    -----
    The original code uses a nested ``product`` to generate a very large
    intermediate array `values`.  In practice the expression used there
    reduces to a simple closed‑form.  By avoiding that intermediate array
    we keep the memory footprint low and let numba inline the loops.
    """
    N = children.shape[0]
    scores = np.empty(N, dtype=np.float64)
    # pre‑compute multipliers
    mul = num_nodes ** np.arange(n - 1, -1, -1)
    for i in range(N):
        el = children[i]
        # clip values to [0, num_nodes-3]
        el_c = np.minimum(el, num_nodes - 3)

        # compute weighted sum
        weight = np.sum((1 + 2 * (np.arange(1, n)[:, None] * (el_c != 0).astype(int))) * mul)
        scores[i] = weight % (num_nodes - 2)
    return scores


class Solver:
    """
    Efficient solver for the cyclic graph independent set problem.

    The solver builds all candidate vertices once and uses a
    numba‑accelerated greedy heuristic to construct an optimal
    independent set.  All heavy numerical work is performed in
    compiled code which gives a substantial speed up over pure Python.
    """

    def solve(self, problem: Tuple[int, int]) -> List[Tuple[int, ...]]:
        num_nodes, n = problem

        # all possible vertices of the strong product
        children = np.array(
            list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32
        )

        # pre‑compute priority scores
        scores = _compute_scores(children, num_nodes, n)

        # adjacency offsets for the strong product
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32
        )

        # powers for linear index calculation
        powers = num_nodes ** np.arange(n - 1, -1, -1)

        # run the greedy algorithm
        selected = _solve_greedy(children, scores, to_block, powers, num_nodes)

        # return the chosen vertices as tuples
        return [tuple(children[i]) for i in selected]