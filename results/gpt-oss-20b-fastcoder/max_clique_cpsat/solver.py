from typing import List

def solve(problem: List[List[int]]) -> List[int]:
    """
    Solve the maximum clique problem using an efficient Bron–Kerbosch algorithm
    with pivoting and bitset manipulation (Python integers). The input is an
    adjacency matrix of a simple undirected graph.

    :param problem: 2D adjacency matrix
    :return: list of vertex indices that form a maximum clique
    """
    n = len(problem)

    # Pre‑compute adjacency bit masks for each vertex.
    adj = [0] * n
    for i in range(n):
        mask = 0
        row = problem[i]
        for j, val in enumerate(row):
            if val:
                mask |= 1 << j
        adj[i] = mask

    # Initialise the best clique discovered so far
    best_clique: int = 0
    best_size: int = 0

    def bronkkerbosch(R: int, P: int, X: int) -> None:
        nonlocal best_clique, best_size
        if P == 0 and X == 0:          # R is a maximal clique
            size = R.bit_count()
            if size > best_size:
                best_size = size
                best_clique = R
            return

        # Choose a pivot u from P ∪ X to reduce recursion
        # Heuristic: pick pivot with maximum degree in P ∪ X
        union = P | X
        if union:
            # find pivot with most neighbours in P
            max_deg = -1
            pivot = 0
            tmp = union
            while tmp:
                u = (tmp & -tmp).bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                tmp &= tmp - 1
            # Candidates: vertices in P that are not neighbours of pivot
            candidates = P & ~adj[pivot]
        else:
            candidates = P

        while candidates:
            v = (candidates & -candidates).bit_length() - 1
            v_bit = 1 << v
            bronkkerbosch(R | v_bit, P & adj[v], X & adj[v])
            P &= ~v_bit
            X |= v_bit
            candidates &= ~v_bit

            # Symmetry breaking: prune if even all remaining vertices
            # can't beat current best
            if R.bit_count() + P.bit_count() <= best_size:
                return

    # Call the algorithm with all vertices in P
    bronkkerbosch(0, (1 << n) - 1, 0)

    # Convert bitmask back to list of indices
    result = []
    mask = best_clique
    while mask:
        v = (mask & -mask).bit_length() - 1
        result.append(v)
        mask &= mask - 1
    return result