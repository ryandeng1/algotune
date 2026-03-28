# noqa
from typing import List

class Solver:
    """
    Fast maximum clique solver based on the Bron–Kerbosch algorithm with pivoting.
    The graph is represented as a list of adjacency lists where the vertices are
    numbered from 0 to n-1.
    The algorithm uses Python integers as bit‑sets for maximum speed.
    """

    def solve(self, problem: List[List[int]]) -> List[int]:
        # Convert adjacency matrix to adjacency bitsets
        n = len(problem)
        if n == 0:
            return []

        # vector of bit masks for each vertex: neighbors (including self)
        neigh = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if problem[i][j]:
                    mask |= 1 << j
            neigh[i] = mask

        # Helper: count bits in an integer (Python 3.10+ has int.bit_count())
        bit_count = int.bit_count

        best_clique: List[int] = []

        # Recursive Bron–Kerbosch with pivot
        def bk(R: int, P: int, X: int):
            nonlocal best_clique
            if P == 0 and X == 0:
                size = bit_count(R)
                if size > len(best_clique):
                    # decode clique
                    clique = [i for i in range(n) if (R >> i) & 1]
                    best_clique = clique
                return

            # Node with maximal degree as pivot to reduce branching
            u = 0
            # choose pivot u from P | X
            combined = P | X
            # heuristic: pick first set bit
            if combined:
                u = (combined & -combined).bit_length() - 1

            # neighbours of pivot
            with_u = neigh[u]
            # non-neighbours of pivot within P
            candidates = P & ~with_u

            while candidates:
                # pick a vertex v (rightmost set bit)
                v = (candidates & -candidates).bit_length() - 1
                v_mask = 1 << v
                # recurse
                bk(R | v_mask, P & neigh[v], X & neigh[v])
                # remove v from P and add to X
                P &= ~v_mask
                X |= v_mask
                # remove v from candidates
                candidates &= ~v_mask

        # Initial call: R empty, P all vertices, X empty
        ALL = (1 << n) - 1
        bk(0, ALL, 0)
        return best_clique