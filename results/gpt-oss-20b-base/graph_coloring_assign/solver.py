def solve(self, problem: list[list[int]]) -> list[int]:
    """
    Fast graph coloring using a DSATUR (Degree of Saturation) heuristic.

    :param problem: 2D adjacency matrix of the graph.
    :return: A list of colors (1‑based) for each vertex.
    """
    n = len(problem)
    # Build adjacency lists
    neigh = [set() for _ in range(n)]
    for i in range(n):
        row = problem[i]
        for j, val in enumerate(row):
            if val and i != j:
                neigh[i].add(j)

    # DSATUR algorithm
    uncolored = set(range(n))
    color_of = [0] * n
    sat_degree = [0] * n          # number of different colors in neighbors
    neighbor_colors = [set() for _ in range(n)]

    while uncolored:
        # choose vertex with highest saturation degree, break ties by highest node degree
        v = max(uncolored, key=lambda x: (sat_degree[x], len(neigh[x])))
        # assign the smallest available color
        used = set(color_of[u] for u in neigh[v] if color_of[u])
        new_color = 1
        while new_color in used:
            new_color += 1
        color_of[v] = new_color
        uncolored.remove(v)

        # update saturation of neighbors
        for u in neigh[v]:
            if color_of[u] == 0:
                if new_color not in neighbor_colors[u]:
                    neighbor_colors[u].add(new_color)
                    sat_degree[u] = len(neighbor_colors[u])

    # remap colors to be consecutive starting at 1
    distinct = sorted(set(color_of))
    remap = {c: i + 1 for i, c in enumerate(distinct)}
    return [remap[c] for c in color_of]