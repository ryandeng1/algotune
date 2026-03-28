import itertools
import numpy as np
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Helper functions (pure NumPy, no external libs for speed)
# --------------------------------------------------------------------------- #

def _priority_vectorized(candidates: np.ndarray, num_nodes: int) -> np.ndarray:
    """Compute the priority score for every candidate vector in a vectorised way."""
    n = candidates.shape[1]
    # Clip values
    clipped = np.minimum(candidates, num_nodes - 3)

    # Generate the “values” array: 2 * product(range(1, n), repeat=n)
    # It corresponds to all combinations of (1..n-1) repeated n times, scaled by 2.
    ranges = np.arange(1, n)
    prod = np.array(list(itertools.product(*([ranges] * n))), dtype=np.int32)
    values = 2 * prod

    multipliers = num_nodes ** np.arange(n - 1, -1, -1, dtype=np.int64)

    # Compute weighted sum for each candidate
    weight = (1 + values + clipped) @ multipliers.astype(np.int64)
    return (weight % (num_nodes - 2)).astype(np.float64).sum(axis=-1)

def _block_conflicts(mask: np.ndarray, candidate: np.ndarray, to_block: np.ndarray, powers: np.ndarray, num_nodes: int):
    """Mask all vertices that would conflict with the chosen candidate."""
    # Compute blocked indices for all shifts simultaneously
    shifted = (candidate + to_block) % num_nodes
    blocked_idx = np.einsum('ij,j->i', shifted, powers)
    mask[blocked_idx] = False

# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #

class Solver:
    def solve(self, problem: Tuple[int, int]) -> List[Tuple[int, ...]]:
        """Fast greedy independent set for n‑th strong product of Cnum_nodes."""
        num_nodes, n = problem

        # All possible n‑tuples
        children = np.array(
            list(itertools.product(range(num_nodes), repeat=n)),
            dtype=np.int32,
        )
        # Pre‑compute shift patterns
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)),
            dtype=np.int32,
        )
        powers = num_nodes ** np.arange(n - 1, -1, -1, dtype=np.int64)

        # Compute priority scores once
        scores = _priority_vectorized(children, num_nodes)

        # Boolean mask of still‑available vertices
        available = np.ones(len(children), dtype=bool)

        selected = []
        while True:
            # Mask out already invalid indices
            scores_available = np.where(available, scores, -np.inf)
            best_idx = int(np.argmax(scores_available))
            best_score = scores_available[best_idx]
            if best_score == -np.inf:
                break

            selected.append(best_idx)
            available[best_idx] = False
            _block_conflicts(available, children[best_idx], to_block, powers, num_nodes)

        return [tuple(children[i]) for i in selected]