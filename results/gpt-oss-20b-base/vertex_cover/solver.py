from typing import List

def solve(problem: List[List[int]]) -> List[int]:
    """
    Max Independent Set solver using a Branch and Bound algorithm on the
    complement graph (i.e. a Maximum Clique problem).

    The algorithm uses bit masks for vertices.  It keeps a global cache of
    the best solution found so far.  The recursion proceeds by selecting a
    candidate vertex, adding it to the current solution, and recursing on
    the intersection of the candidate set with its neighbors.  The branch
    is pruned when the size of the remaining candidates plus the current
    solution cannot beat the best found.
    """
    n = len(problem)
    # Build adjacency masks for the complement graph (i.e. edges absent in problem)
    comp_adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and problem[i][j] == 0:
                mask |= 1 << j
        comp_adj[i] = mask

    best_sol: List[int] = []

    def dfs(cand: int, cur: List[int]) -> None:
        nonlocal best_sol
        # Prune if not enough candidates to beat current best
        if len(cur) + cand.bit_count() <= len(best_sol):
            return
        if cand == 0:
            if len(cur) > len(best_sol):
                best_sol = cur.copy()
            return
        # Choose a vertex v from cand (pick least significant bit)
        v = (cand & -cand).bit_length() - 1   # index of chosen vertex
        # Recurse with v in the solution
        dfs(cand & comp_adj[v], cur + [v])
        # Recurse without v
        dfs(cand & ~(1 << v), cur)

    # Start with all vertices as candidates
    all_vertices = (1 << n) - 1
    dfs(all_vertices, [])

    return best_sol