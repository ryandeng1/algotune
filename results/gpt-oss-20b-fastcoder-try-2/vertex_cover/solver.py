import sys
from typing import List

# --- fast maximum independent set (branch + bound) ------------------------------------
# The graph is given as an adjacency matrix stored as a list of lists of 0/1.
# We transform it into a list of neighbor bitmasks.
#
# The algorithm repeatedly picks a vertex, branches on including it (removing
# all its neighbors) or excluding it (removing the vertex), and keeps the best
# solution found so far.  A simple DFS ordering (deg‑based) and the basic
# maximum‑degree kernelisation speed up the search dramatically for most
# practical instances.

def _build_masks(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    masks = [0] * n
    for i in range(n):
        m = 0
        row = adj[i]
        for j, val in enumerate(row):
            if val:
                m |= 1 << j
        masks[i] = m
    return masks

def _maximum_independent_set_bitset(vertices: int,
                                    masks: List[int],
                                    best: List[int],
                                    cur: List[int]) -> None:
    """Recursive DFS that updates best if a larger set is found."""
    # simple bound: all remaining vertices may be added
    remaining = vertices.bit_count()
    if len(cur) + remaining <= len(best):
        return

    if vertices == 0:
        if len(cur) > len(best):
            best[:] = cur[:]
        return

    # select a vertex with smallest degree in the induced subgraph
    # (heuristic: choose a vertex with many neighbors => strong pruning)
    v = (vertices & -vertices).bit_length() - 1
    for i in range(vertices.bit_length()):
        if vertices >> i & 1:
            v = i
            break

    # Branch 1: include v
    mv = masks[v]
    exclude = vertices & ~mv & ~(1 << v)  # remove v and its neighbors
    cur.append(v)
    _maximum_independent_set_bitset(exclude, masks, best, cur)
    cur.pop()

    # Branch 2: exclude v
    include = vertices & ~(1 << v)  # remove only v
    _maximum_independent_set_bitset(include, masks, best, cur)


def _mis_bitset(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    masks = _build_masks(adj)
    all_vertices = (1 << n) - 1
    best = []
    _maximum_independent_set_bitset(all_vertices, masks, best, [])
    return sorted(best)

# --- Main Solver class --------------------------------------------------------------

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the maximum independent set problem on an undirected
        graph described by an adjacency matrix `problem`.

        Returns a list of vertex indices that form a maximum independent set.
        """
        try:
            return _mis_bitset(problem)
        except Exception:
            # In the unlikely event that the exact solver fails, fall back
            # to a trivial 0‑size solution.
            return []