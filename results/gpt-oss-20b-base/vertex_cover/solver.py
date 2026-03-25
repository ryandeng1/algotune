#!/usr/bin/env python3
# solver.py

import sys
from typing import List, Any, Set

class Solver:
    """Fast exact solver for the Vertex Cover problem.
    
    The algorithm finds a maximum independent set via a branch‑and‑bound
    recursion using bitmasks (works efficiently for up to ~60 vertices).
    The optimal vertex cover is the complement of that independent set.
    """

    def __init__(self) -> None:
        # Upper bound for the size of a maximum independent set.
        # We will gradually improve this bound during the search.
        self.best_independent: int = 0
        self.best_mask: int = 0

    def _neighbor_mask(self, v: int, adj: List[int]) -> int:
        """Return a bitmask of neighbors of vertex v (including v itself)."""
        return adj[v] | (1 << v)

    def _branch_and_bound(self, candidates: int, adj: List[int]) -> None:
        """
        Recursive search for MIS.
        `candidates` is a bitmask of vertices that can still be added.
        """
        # If no candidates left, update best solution
        if candidates == 0:
            size = self._popcount(self.best_mask)
            if size > self._popcount(self.best_mask):
                self.best_mask = self.best_mask
                self.best_independent = size
            return

        # Simple bound: if current independent set size + remaining vertices
        # is <= best found, cut branch
        current_size = self._popcount(self.best_mask)
        if current_size + self._popcount(candidates) <= self.best_independent:
            return

        # Choose a vertex with maximum degree among candidates (heuristic)
        v = None
        max_deg = -1
        tmp = candidates
        while tmp:
            u = tmp & -tmp
            idx = u.bit_length() - 1
            deg = self._popcount(adj[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = idx
            tmp -= u

        # Branch 1: include v
        new_candidates = candidates & ~self._neighbor_mask(v, adj)
        self.best_mask |= (1 << v)
        self._branch_and_bound(new_candidates, adj)
        self.best_mask &= ~(1 << v)

        # Branch 2: exclude v
        new_candidates = candidates & ~(1 << v)
        self._branch_and_bound(new_candidates, adj)

    @staticmethod
    def _popcount(x: int) -> int:
        """Return number of set bits in integer x."""
        return x.bit_count()

    def _max_independent_set(self, adj: List[int]) -> List[int]:
        """Return a list of vertices forming a maximum independent set."""
        n = len(adj)
        all_mask = (1 << n) - 1
        self.best_mask = 0
        self.best_independent = 0
        self._branch_and_bound(all_mask, adj)
        mask = self.best_mask
        result = []
        idx = 0
        while mask:
            lsb = mask & -mask
            result.append(idx)
            mask -= lsb
            idx += 1
        return result

    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """Return a minimum vertex cover for the given adjacency matrix."""
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency bitmask for each vertex
        adj = [0] * n
        for i in range(n):
            row = problem[i]
            mask = 0
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            adj[i] = mask

        # Find maximum independent set (MIS)
        mis = self._max_independent_set(adj)

        # Vertex cover is complement of MIS
        cover = [i for i in range(n) if i not in mis]
        return cover
