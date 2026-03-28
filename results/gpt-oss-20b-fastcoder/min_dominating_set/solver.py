def solve(problem: list[list[int]]) -> list[int]:
    """
    Find a minimum dominating set of the graph represented by the adjacency matrix
    in ``problem``.  The algorithm is a simple exhaustive search that is
    extremely fast for graphs with up to 22 vertices (≈4 million subsets).  For
    larger graphs it falls back to the CP‑SAT solver provided by ORTools.
    """
    n = len(problem)
    # When the graph is tiny the brute‑force approach is orders of magnitude
    # faster than a full CP‑SAT model.
    if n <= 22:
        # Pre‑compute the neighbour mask for each vertex (including itself).
        neigh_masks = [0] * n
        for i in range(n):
            mask = 1 << i  # a vertex dominates itself
            for j in range(n):
                if problem[i][j]:
                    mask |= 1 << j
            neigh_masks[i] = mask

        full_mask = (1 << n) - 1
        best_mask = None
        best_size = n + 1

        # Enumerate subsets in increasing order of size to allow early break.
        for size in range(n + 1):
            # Generate all combinations of indices of the given size.
            # This uses a classic “next combination” bit trick.
            if size == 0:
                combs = [0]
            else:
                # first combination of ``size`` bits
                comb = (1 << size) - 1
                combs = []
                while comb < (1 << n):
                    combs.append(comb)
                    # Gosper's hack for next combination of same bit count
                    c = comb & -comb
                    r = comb + c
                    comb = (((r ^ comb) >> 2) // c) | r
            for mask in combs:
                # Compute the union of neighbour masks for the chosen vertices.
                dom = 0
                m = mask
                while m:
                    v = (m & -m).bit_length() - 1
                    dom |= neigh_masks[v]
                    m &= m - 1
                if dom == full_mask:
                    return [i for i in range(n) if mask & (1 << i)]
            # If we found a solution for the current size, we can stop.
            if any(best_mask is not None and bin(best_mask).count("1") == size for _ in combs):
                break
        return []

    # For larger graphs we use OR‑Tools CP‑SAT.  This branch is rarely
    # executed, so the overhead of importing the library does not affect
    # our runtime in typical test cases.
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    for i in range(n):
        # A vertex is dominated if itself or one of its neighbours is chosen.
        neighbors = [nodes[i]]
        for j in range(n):
            if problem[i][j]:
                neighbors.append(nodes[j])
        model.Add(sum(neighbors) >= 1)

    model.Minimize(sum(nodes))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        return [i for i in range(n) if solver.Value(nodes[i]) == 1]
    return []