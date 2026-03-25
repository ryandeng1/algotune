# solver.py
import math
from itertools import combinations
from typing import List, Any

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Find a minimum dominating set of an undirected graph represented by an
        adjacency matrix.  The input matrix is symmetric with entries 0/1.
        For graphs with up to ~25 vertices this exhaustive bit‑mask approach is
        fast enough and guarantees optimality.

        :param problem: 2D adjacency matrix of the graph.
        :return: List of vertex indices forming a minimum dominating set.
        """
        n = len(problem)
        # Precompute neighbor masks (including self)
        neigh_masks = [0] * n
        for i in range(n):
            mask = 1 << i
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            neigh_masks[i] = mask

        # Target cover mask (all vertices)
        full_mask = (1 << n) - 1

        # Try subsets by increasing size
        for size in range(1, n + 1):
            # Generate all combinations of vertices of current size
            for combo in combinations(range(n), size):
                cover = 0
                for v in combo:
                    cover |= neigh_masks[v]
                if cover == full_mask:
                    return list(combo)
        # If no dominating set found (shouldn't happen on valid input)
        return []
