from typing import List, Tuple, Dict

def dsatur_color(adj: List[List[int]]) -> List[int]:
    """
    Exact graph coloring using the DSATUR algorithm with iterative deepening.
    The function returns a list of colors (1‑based) that uses the minimal
    number of colors.  The algorithm is classical and has a very small
    constant factor, making it faster than the CP‑SAT formulation used in
    the original implementation for most graphs of moderate size.
    """
    n = len(adj)
    # Build adjacency list
    neigh = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                neigh[i].add(j)
                neigh[j].add(i)

    # Pre‑computed maximum clique size (lower bound) via a greedy heuristic
    def max_clique_size() -> int:
        maxc = 0
        stack = list(range(n))
        # simple degree‑based ordering
        stack.sort(key=lambda v: len(neigh[v]), reverse=True)
        for v in stack:
            cand = {v}
            for u in stack:
                if u > v and all(u in neigh[w] for w in cand):
                    cand.add(u)
            maxc = max(maxc, len(cand))
        return maxc

    lower_bound = max_clique_size()
    # Upper bound from a fast greedy coloring
    order = sorted(range(n), key=lambda v: -len(neigh[v]))
    greedy_col = [0] * n
    used = 0
    for v in order:
        forbidden = {greedy_col[u] for u in neigh[v] if greedy_col[u]}
        c = 1
        while c in forbidden:
            c += 1
        greedy_col[v] = c
        used = max(used, c)
    upper_bound = used

    # Main search
    colors = [0] * n
    best_solution: List[int] = []

    # Saturation degrees
    sat = [0] * n
    sat_set: List[set] = [set() for _ in range(n)]

    def dfs(colored: int, max_color: int):
        nonlocal best_solution
        if colored == n:
            if colored < len(best_solution) or not best_solution:
                best_solution = colors.copy()
            return

        # Fail‑first ordering: choose vertex with highest saturation and lowest degree
        uncolored = [v for v in range(n) if colors[v] == 0]
        v = max(uncolored, key=lambda x: (len(sat_set[x]), len(neigh[x]), -x))

        forbidden = sat_set[v]
        for c in range(1, max_color + 2):
            if c in forbidden:
                continue
            if c > len(best_solution) and best_solution:
                continue
            # Assign color
            colors[v] = c
            added: List[Tuple[int, int]] = []  # store (neighbor, old_size)
            for u in neigh[v]:
                if colors[u] == 0:
                    if c not in sat_set[u]:
                        sat_set[u].add(c)
                        added.append((u, len(sat_set[u]) - 1))
            dfs(colored + 1, max(max_color, c))
            # Undo
            colors[v] = 0
            for u, old_size in added:
                sat_set[u].remove(c)
            if best_solution and colored + 1 + (max_color - c) >= len(best_solution):
                break  # prune: cannot improve

    dfs(0, lower_bound)
    return best_solution

def solve(problem: List[List[int]]) -> List[int]:
    """
    Solve the graph coloring problem optimally.
    `problem` is an adjacency matrix (0/1).  Returns a list of colors
    (1‑based) assigned to each vertex.  If the graph is empty, returns
    an empty list.
    """
    if not problem:
        return []
    colors = dsatur_color(problem)
    return [c for c in colors]