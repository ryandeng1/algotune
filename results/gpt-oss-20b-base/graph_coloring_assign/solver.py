def solve(problem: list[list[int]]) -> list[int]:
    """
    Fast heuristic graph coloring for a 0-1 adjacency matrix.

    The algorithm uses the Welsh–Powell ordering (descending degree)
    followed by a basic greedy color assignment.  This is much faster
    than the previous CP‑SAT approach while still producing a valid
    coloring using a small number of colors.

    :param problem: 2‑D list representing the adjacency matrix of a graph
    :return: List of colors (1‑based) for each vertex
    """
    n = len(problem)
    if n == 0:
        return []

    # Build adjacency lists and compute degrees
    adjacency: list[list[int]] = [[] for _ in range(n)]
    degree = [0] * n
    for i in range(n):
        row = problem[i]
        for j in range(i + 1, n):
            if row[j]:
                adjacency[i].append(j)
                adjacency[j].append(i)
                degree[i] += 1
                degree[j] += 1

    # Welsh–Powell ordering: sort by decreasing degree
    order = sorted(range(n), key=lambda v: -degree[v])

    colors = [0] * n
    for v in order:
        # Determine colors used by neighbors
        used = 0
        for u in adjacency[v]:
            c = colors[u]
            if c:
                used |= 1 << (c - 1)

        # Find the first available color (smallest unused)
        color = 1
        mask = 1
        while used & mask:
            color += 1
            mask <<= 1
        colors[v] = color

    return colors