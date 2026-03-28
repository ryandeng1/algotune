import itertools
import numpy as np
from typing import List, Tuple


def solve(problem: Tuple[int, int]) -> List[Tuple[int, ...]]:
    """
    Compute an optimal independent set in the n‑th strong product of a cycle
    graph with `num_nodes` vertices. The algorithm is a greedy, vectorized
    implementation of the score‑based selection described in the original
    solution.

    This implementation uses only NumPy (no Numba) and is fully vectorized,
    avoiding explicit Python loops wherever possible. It is therefore
    significantly faster for large instances.
    """
    num_nodes, n = problem

    # All candidate vertices: (num_nodes ** n, n) array
    all_vertices = np.array(
        list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int64
    )

    # Pre‑compute the priority scores for all vertices
    #  values: (n‑tuple of numbers 1..n‑1) once
    vals = np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int64)
    # multipliers for weighted sum
    mults = (num_nodes ** np.arange(n - 1, -1, -1)).astype(np.int64)
    # repeated for each vertex
    vals_matrix = np.tile(vals, (all_vertices.shape[0], 1))
    # clip vertices to [0, num_nodes-3]
    clipped = np.clip(all_vertices, 0, num_nodes - 3, dtype=np.int64)
    weighted = np.sum((1 + vals_matrix + clipped) * mults, axis=-1)
    scores = np.sum(weighted % (num_nodes - 2), dtype=np.float64)

    # Prepare the blocking offsets: all n‑tuples of -1,0,1
    block_offsets = np.array(
        list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int64
    )

    # Helper to convert n‑tuple index to integer (base num_nodes)
    powers = (num_nodes ** np.arange(n - 1, -1, -1)).astype(np.int64)

    # Boolean mask of available vertices
    avail = np.ones(all_vertices.shape[0], dtype=bool)

    selected = []

    while True:
        if not avail.any():
            break
        # choose the available vertex with max score
        avail_scores = np.where(avail, scores, -np.inf)
        best_idx = int(np.argmax(avail_scores))
        if avail_scores[best_idx] == -np.inf:  # no more available
            break

        selected.append(best_idx)
        # Block neighbors
        candidate = all_vertices[best_idx]
        # Compute indices of blocked vertices
        # All neighbors = candidate + offsets mod num_nodes
        neighbors = (candidate + block_offsets) % num_nodes
        # Convert each neighbor tuple to integer index
        neigh_idx = np.sum(neighbors[..., None] * powers, axis=1)
        avail[neigh_idx] = False
        avail[best_idx] = False  # block itself as well

    return [tuple(all_vertices[i]) for i in selected]