import math
from functools import lru_cache

class Solver:
    """
    Exact solver for the Maximum Weighted Independent Set problem.
    Uses a fast branch-and-bound algorithm with bitset representation.
    Works efficiently for graphs up to ~70 vertices.
    """
    def solve(self, problem: dict[str, list]):
        adj = problem["adj_matrix"]
        w = problem["weights"]
        n = len(adj)

        # Build neighbor bitmask for each node
        neigh = [0] * n
        for i in range(n):
            mask = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    mask |= 1 << j
            neigh[i] = mask

        # Precompute sorted nodes by degree for good branching
        nodes = list(range(n))
        nodes.sort(key=lambda x: bin(neigh[x]).count("1"), reverse=True)

        # Precompute prefix sums of weights in sorted order for upper bound
        sorted_weights = [w[i] for i in nodes]
        pref_sum = [0]
        for val in sorted_weights:
            pref_sum.append(pref_sum[-1] + val)
        total_weight = pref_sum[-1]

        best_weight = 0
        best_set = 0

        @lru_cache(maxsize=None)
        def dfs(cand_mask: int, weight: int, idx: int):
            nonlocal best_weight, best_set

            # Upper bound pruning
            remaining = total_weight - pref_sum[idx]
            if weight + remaining <= best_weight:
                return

            # Choose next node to branch on
            while idx < n and not (cand_mask & (1 << nodes[idx])):
                idx += 1

            if idx == n:
                if weight > best_weight:
                    best_weight = weight
                    best_set = cand_mask
                return

            v = nodes[idx]
            mask_v = cand_mask & ~((1 << v) | neigh[v])

            # Branch 1: include v
            dfs(mask_v, weight + w[v], idx + 1)

            # Branch 2: exclude v
            dfs(cand_mask & ~(1 << v), weight, idx + 1)

        # Start with all nodes available
        dfs((1 << n) - 1, 0, 0)

        # Translate bitmask back to indices
        result = [i for i in range(n) if best_set & (1 << i)]
        return result
