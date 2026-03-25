import sys
from typing import List, Any

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the maximum clique problem using a branch-and-bound
        implementation of the Bron–Kerbosch algorithm with pivoting
        and bitset optimisations.
        """
        n = len(problem)
        if n == 0:
            return []

        # Represent adjacency as bit masks (int). 1 << i means node i is present.
        adj = [0] * n
        for i in range(n):
            mask = 0
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            adj[i] = mask

        # Initial candidate set: all vertices.
        all_vertices = (1 << n) - 1

        best_clique: int = 0  # bitmask of best clique found
        best_size: int = 0

        def bronk(R: int, P: int, X: int):
            nonlocal best_clique, best_size
            # If both P and X are empty, R is a maximal clique
            if P == 0 and X == 0:
                size = R.bit_count()
                if size > best_size:
                    best_size = size
                    best_clique = R
                return

            # Bounding: if size of current clique plus remaining candidates <= best, prune
            if R.bit_count() + P.bit_count() <= best_size:
                return

            # Choose pivot u from P ∪ X that maximises |P ∩ N(u)|
            u = None
            max_deg = -1
            pu = P | X
            while pu:
                v = (pu & -pu).bit_length() - 1
                pu &= pu - 1
                deg = (P & adj[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    u = v

            # Candidates: P \ N(u)
            if u is not None:
                candidates = P & ~adj[u]
            else:
                candidates = P

            while candidates:
                v = (candidates & -candidates).bit_length() - 1
                candidates &= candidates - 1
                R_new = R | (1 << v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bronk(R_new, P_new, X_new)
                P &= ~(1 << v)
                X |= (1 << v)

        bronk(0, all_vertices, 0)

        # Convert bitmask to list of indices
        result = []
        mask = best_clique
        while mask:
            v = (mask & -mask).bit_length() - 1
            result.append(v)
            mask &= mask - 1
        result.sort()
        return result
