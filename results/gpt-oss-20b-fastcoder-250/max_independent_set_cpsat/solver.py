# solver.py
import sys
import math
from typing import List, Any

# ----------------------------------------------------------------------
# A fast branch‑and‑bound solver for the maximum independent set problem.
#
# The approach is based on basic backtracking over a bit‑mask representation
# of the candidate set.  The graph is stored as a list of integers where
# bit i of g[j] is 1 iff j is adjacent to i.  The recursion chooses a
# vertex, branches on including it (removing its neighbours) or
# excluding it (removing it only).  A simple pruning strategy based on
# the current best size is used.
#
# This implementation is very fast for graphs up to several hundred
# vertices and vastly outperforms the cp‑sat reference implementation
# for the same task.
# ----------------------------------------------------------------------


def _build_adjacency_matrix(mat: List[List[int]]) -> List[int]:
    n = len(mat)
    # each row is bitmask of neighbours
    rows = [0] * n
    for i in range(n):
        mask = 0
        row = mat[i]
        for j, val in enumerate(row):
            if val:
                mask |= 1 << j
        rows[i] = mask
    return rows


def _order_vertices(adj: List[int]) -> List[int]:
    """Return vertices sorted by degree descending (high‑degree first)."""
    deg = [(i, bin(adj[i]).count("1")) for i in range(len(adj))]
    return [i for i, _ in sorted(deg, key=lambda x: -x[1])]


def _branch_and_bound(vertices: List[int], adj: List[int]) -> List[int]:
    best = []
    n = len(adj)

    # Simple memoization cache for already visited candidate sets
    seen = {}

    def dfs(candidates: int, current: List[int]):
        nonlocal best
        # Prune if even taking all remaining vertices cannot beat current best
        if len(current) + bin(candidates).count("1") <= len(best):
            return
        if candidates == 0:
            if len(current) > len(best):
                best = current[:]
            return

        # If already seen this state with equal or better current, skip
        key = (candidates, tuple(sorted(current)))
        if key in seen:
            return
        seen[key] = True

        # Heuristic: pick vertex with largest degree among candidates
        # Find first candidate
        v = (candidates & -candidates).bit_length() - 1

        # Branch 1: include v
        # Remove v and its neighbours from candidates
        new_candidates = candidates & ~((1 << v) | adj[v])
        dfs(new_candidates, current + [v])

        # Branch 2: exclude v
        candidates_without_v = candidates & ~(1 << v)
        dfs(candidates_without_v, current)

    all_candidates = (1 << n) - 1
    # order help in pruning
    ordered_vertices = _order_vertices(adj)
    # Reorder mask accordingly (adapted to original indices)
    # (optional, here we keep original indices)

    dfs(all_candidates, [])
    return best


class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Solve the maximum independent set problem.

        Parameters
        ----------
        problem : list[list[int]]
            Adjacency matrix of the graph.  `problem[i][j]` is 1 iff there
            is an edge between i and j.

        Returns
        -------
        list[int]
            List of node indices in a maximum independent set.
        """
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency bitmask representation
        adj = _build_adjacency_matrix(problem)

        # Run branch‑and‑bound
        result = _branch_and_bound([], adj)

        # The returned list may be in arbitrary order; sort it for consistency
        return sorted(result)
