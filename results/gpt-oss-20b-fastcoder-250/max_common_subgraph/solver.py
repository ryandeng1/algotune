#!/usr/bin/env python3
# solver.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        """
        Find a maximum common induced subgraph between two unlabelled undirected graphs.
        The algorithm constructs a compatibility graph on all possible pair mappings
        and searches for its maximum clique using a Bron–Kerbosch algorithm with
        pivoting and bitset optimisation.
        """
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)

        # Build list of vertices (i,p) and a mapping from vertex id to pair
        vertices: List[Tuple[int, int]] = [(i, p) for i in range(n) for p in range(m)]
        V = len(vertices)
        if V == 0:
            return []

        # Pre‑compute compatibility bitmask for each vertex
        # bit k is 1 if vertex k is compatible with current vertex
        compat = [0] * V
        for idx1, (i, p) in enumerate(vertices):
            mask = 0
            for idx2, (j, q) in enumerate(vertices):
                if idx1 == idx2:
                    continue
                if A[i][j] == B[p][q]:
                    mask |= 1 << idx2
            compat[idx1] = mask

        # Ordering vertices by degeneracy to improve speed
        order = list(range(V))
        # simple heuristic: sort by degree descending
        degrees = [bin(compat[v]).count("1") for v in range(V)]
        order.sort(key=lambda v: -degrees[v])

        # Global best
        best_clique_mask = 0
        best_size = 0

        # Bitset utilities
        def popcnt(x: int) -> int:
            return x.bit_count()

        # Bron–Kerbosch with pivot
        def bk(R: int, P: int, X: int) -> None:
            nonlocal best_clique_mask, best_size
            if P == 0 and X == 0:
                sz = popcnt(R)
                if sz > best_size:
                    best_size = sz
                    best_clique_mask = R
                return
            # Upper bound pruning
            if popcnt(R) + popcnt(P) <= best_size:
                return
            # Pivot selection: choose vertex u in P ∪ X maximizing |P ∩ N(u)|
            u = -1
            max_deg = -1
            PX = P | X
            while PX:
                v = (PX & -PX).bit_length() - 1
                PX &= PX - 1
                deg = popcnt(P & compat[v])
                if deg > max_deg:
                    max_deg = deg
                    u = v
            # Explore candidates not adjacent to pivot
            candidates = P & ~compat[u]
            while candidates:
                v = (candidates & -candidates).bit_length() - 1
                candidates &= candidates - 1
                R_new = R | (1 << v)
                P_new = P & compat[v]
                X_new = X & compat[v]
                bk(R_new, P_new, X_new)
                P &= ~(1 << v)
                X |= (1 << v)

        # Initial call
        all_vertices_mask = (1 << V) - 1
        bk(0, all_vertices_mask, 0)

        # Decode best clique mask into mapping pairs
        result: List[Tuple[int, int]] = []
        mask = best_clique_mask
        while mask:
            v = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            result.append(vertices[v])
        result.sort()
        return result
