import itertools
import numpy as np
from typing import List, Tuple

class Solver:
    def solve(self, problem: Tuple[int, int]) -> List[Tuple[int, ...]]:
        num_nodes, n = problem

        # All vertices of the n‑fold product (cyclic graph)
        vertices = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        N = vertices.shape[0]
        # All displacement vectors in {‑1,0,1}^n that can block
        to_block = np.array(list(itertools.product((-1, 0, 1), repeat=n)), dtype=np.int32)

        # Pre–compute powers of the base number of nodes for linear indexing
        powers = np.array([num_nodes ** i for i in range(n - 1, -1, -1)], dtype=np.int32)

        # Compute initial priority scores in a single vectorised pass
        # We use the same formula as priority() but fully vectorised
        # Clip values
        clipped = np.clip(vertices, None, num_nodes - 3)
        # 2 * (product of range(1, n)) repeated for each vertex
        # generate the weight matrix for the 2*values part
        base_values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int32)
        # Weighted sum
        weighted = (1 + base_values + clipped) @ powers
        scores = np.sum(weighted % (num_nodes - 2), axis=1, dtype=np.float64)

        # Boolean mask for vertices still considered
        active = np.ones(N, dtype=bool)
        selected_idx = []

        # Greedy loop
        while True:
            # Mask scores of inactive vertices
            masked_scores = np.where(active, scores, -np.inf)
            best_idx = int(np.argmax(masked_scores))
            best_score = masked_scores[best_idx]
            if best_score == -np.inf:
                break
            selected_idx.append(best_idx)
            # Block all vertices reachable by adding any displacement in to_block
            # Compute indices to block
            affected = vertices[best_idx] + to_block  # shape (3^n, n)
            # Modulo wrap-around
            affected_mod = np.mod(affected, num_nodes)
            # Convert to linear index
            blocked = (affected_mod @ powers).astype(np.int32)
            active[blocked] = False

        return [tuple(vertices[i]) for i in selected_idx]

    # The helper `_priority` is no longer needed because the scoring is fully vectorised