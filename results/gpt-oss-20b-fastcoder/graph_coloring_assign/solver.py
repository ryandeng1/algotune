from ortools.sat.python import cp_model
from itertools import combinations

def solve(problem):
    n = len(problem)
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if problem[i][j]:
                adj[i].add(j)
                adj[j].add(i)

    # Quick lower bound: size of a maximal clique via greedy heuristic
    def greedy_clique():
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        clique = set()
        for v in order:
            if all(v in adj[u] for u in clique):
                clique.add(v)
        return clique

    lower = len(greedy_clique())

    # Upper bound: greedy coloring (largest first)
    color = {}
    for v in sorted(range(n), key=lambda x: len(adj[x]), reverse=True):
        used = {color[u] for u in adj[v] if u in color}
        c = 1
        while c in used:
            c += 1
        color[v] = c
    upper = max(color.values())

    if lower == upper:
        # Already optimal
        return [color[v] for v in range(n)]

    # Reduce graph by removing dominated vertices
    def reduce_graph():
        dom = {v: v for v in range(n)}
        nodes = set(range(n))
        while True:
            changed = False
            adj_sets = {v: adj[v].intersection(nodes) for v in nodes}
            rem = set()
            for u, v in combinations(nodes, 2):
                if adj_sets[u] <= adj_sets[v]:
                    rem.add(u)
                    dom[u] = v
                elif adj_sets[v] <= adj_sets[u]:
                    rem.add(v)
                    dom[v] = u
            if not rem:
                break
            nodes -= rem
            changed = True
            if not changed:
                break
        return nodes, dom

    nodes, dom = reduce_graph()
    # CP-SAT model
    model = cp_model.CpModel()
    vars = {(u, c): model.NewBoolVar(f'x_{u}_{c}') for u in nodes for c in range(upper)}
    used = [model.NewBoolVar(f'w_{c}') for c in range(upper)]

    # Each vertex chooses one color
    for u in nodes:
        model.AddExactlyOne(vars[u, c] for c in range(upper))

    # Clique seeding: vertices in lower bound clique get fixed colors
    clique = greedy_clique()
    for idx, v in enumerate(sorted(clique)):
        model.Add(vars[v, idx] == 1)

    # Adjacency constraints
    for u, v in combinations(nodes, 2):
        if v in adj[u]:
            for c in range(upper):
                model.Add(vars[u, c] + vars[v, c] <= used[c])

    # Link color usage
    for c in range(upper):
        model.Add(used[c] == sum(vars[u, c] for u in nodes))

    # Monotonicity to reduce search space
    for c in range(1, upper):
        model.Add(used[c - 1] >= used[c])

    model.Minimize(sum(used))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        return []

    # Retrieve coloring from reduced graph
    red_color = {}
    for u in nodes:
        for c in range(upper):
            if solver.Value(vars[u, c]):
                red_color[u] = c + 1
                break

    # Expand to original vertices
    final_color = [0] * n
    for v in range(n):
        root = v
        while dom[root] != root:
            root = dom[root]
        final_color[v] = red_color.get(root, 1)

    # Relabel colors to be 1..k
    used_colors = sorted(set(final_color))
    remap = {old: new for new, old in enumerate(used_colors, start=1)}
    return [remap[c] for c in final_color]