import math
from typing import List, Any

# ----------------------------------------------------------------------
# Fast maximum clique solver using a bit‑set implementation of
# Bron–Kerbosch with pivot and degeneracy ordering.
# ----------------------------------------------------------------------


def _degeneracy_order(adj: List[int]) -> List[int]:
    """Return vertices ordered by degeneracy (Peel order)."""
    n = len(adj)
    degrees = [adj[i].bit_count() for i in range(n)]
    order = []
    used = [False]*n
    for _ in range(n):
        # pick vertex with minimal degree among unused
        min_deg = math.inf
        v = -1
        for i in range(n):
            if not used[i] and degrees[i] < min_deg:
                min_deg = degrees[i]
                v = i
        used[v] = True
        order.append(v)
        mask = adj[v]
        # decrement degrees of neighbors
        while mask:
            w = (mask & -mask).bit_length()-1
            if not used[w]:
                degrees[w] -= 1
            mask &= mask-1
    return order


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """Return a maximum clique of the graph described by `problem`."""
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency bit masks
        adj = [0]*n
        for i in range(n):
            mask = 0
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            adj[i] = mask

        # Degeneracy ordering to speed up Bron‑Kerbosch
        order = _degeneracy_order(adj)
        best_clique = 0
        best_size = 0

        def bk(R: int, P: int, X: int):
            nonlocal best_clique, best_size
            if P == 0 and X == 0:
                # Found a maximal clique
                sz = R.bit_count()
                if sz > best_size:
                    best_size = sz
                    best_clique = R
                return
            # Pivot choice: vertex in P∪X with max degree in P
            U = P | X
            # Choose pivot u with most neighbors in P
            max_deg = -1
            u = -1
            mask = U
            while mask:
                v = (mask & -mask).bit_length()-1
                deg = (adj[v] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    u = v
                mask &= mask-1
            # candidates = P \ N(u)
            candidates = P & ~adj[u]
            mask = candidates
            while mask:
                v = (mask & -mask).bit_length()-1
                mask &= mask-1
                bv = 1 << v
                bk(R | bv, P & adj[v], X & adj[v])
                P &= ~bv
                X |= bv
                # pruning: if current size + remaining P can't beat best
                if R.bit_count() + P.bit_count() <= best_size:
                    return

        # Main loop over degeneracy order
        V_all = (1 << n) - 1
        for v in order:
            bv = 1 << v
            R = bv
            P = adj[v] & (~((1 << v) - 1))  # only vertices after v in order
            X = 0
            bk(R, P, X)
            # remove v from the graph
            V_all &= ~bv

        # Convert best_clique mask to list of indices
        result = []
        mask = best_clique
        while mask:
            v = (mask & -mask).bit_length()-1
            result.append(v)
            mask &= mask-1
        return result
