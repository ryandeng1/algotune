from typing import Any
import numpy as np
from numba.typed import List

def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    n = children.shape[1]
    N = children.shape[0]
    result = List()
    while True:
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        if best_score == -np.inf:
            break
        result.append(best_idx)
        candidate = children[best_idx]
        shifted = (candidate[None, :] + to_block) % num_nodes
        blocked_indices = np.dot(shifted, powers)
        scores[blocked_indices] = -np.inf
    return result