import itertools
from ortools.sat.python import cp_model
from collections import defaultdict, deque

def greedy_color(adj, nodes=None, strategy="largest_first"):
    """A very small greedy coloring implementation."""
    if nodes is None:
        nodes = list(adj.keys())
    # ordering
    if strategy == "largest_first":
        nodes.sort(key=lambda v: -len(adj[v]))
    colors = {}
    for v in nodes:
        used = {colors[u] for u in adj[v] if u in colors}
        c = 1
        while c in used:
            c += 1
        colors[v] = c
    return colors

def max_clique(adj, nodes):
    """simple exact max clique for small graphs."""
    best = []
    for comb in itertools.chain.from_iterable(
            itertools.combinations(nodes, r) for r in range(len(nodes)+1)
        ):
        if not comb:
            continue
        if all((b in adj[a]) for a in comb for b in comb if a < b):
            if len(comb) > len(best):
                best = list(comb)
    return best

def dominance_preprocess(adj):
    """Reduces graph by removing dominated vertices.
       Returns reduced adjacency and dominator map (root->root)."""
    G = {v: set(adj[v]) for v in adj}
    dominator = {v: v for v in G}
    while True:
        to_remove = set()
        items = list(G.items())
        for i in range(len(items)):
            u, au = items[i]
            if u not in G:
                continue
            for j in range(i+1, len(items)):
                v, av = items[j]
                if v not in G:
                    continue
                if au <= av:
                    to_remove.add(u)
                    dominator[u] = v
                elif av <= au:
                    to_remove.add(v)
                    dominator[v] = u
        if not to_remove:
            break
        for v in to_remove:
            G.pop(v, None)
            for u in G:
                G[u].discard(v)
    return G, dominator

class Solver:
    def solve(self, problem):
        n = len(problem)

        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for i in range(n):
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    adj[i].add(j)
                    adj[j].add(i)

        # ---- 1. Reduce graph by dominance ----
        red_adj, dom = dominance_preprocess(adj)

        red_nodes = list(red_adj)
        red_edges = [(u, v) for u in red_nodes for v in red_adj[u] if v > u]

        # ---- 2. Upper bound (greedy) ----
        greedy = greedy_color(red_adj, red_nodes, strategy="largest_first")
        ub = max(greedy.values())

        # ---- 3. Max clique for lower bound ----
        clique = max_clique(red_adj, red_nodes)
        lb = len(clique)

        # ---- 4. If greedy already optimal ----
        if lb == ub:
            full = greedy_color(adj, strategy="largest_first")
            return [full[i]+1 for i in range(n)]

        # ---- 5. CP-SAT model ----
        H = ub
        model = cp_model.CpModel()

        # variables
        x = {(u, c): model.NewBoolVar(f"x_{u}_{c}") for u in red_nodes for c in range(H)}
        w = [model.NewBoolVar(f"w_{c}") for c in range(H)]

        # clique seeding
        for idx, u in enumerate(clique):
            model.Add(x[(u, idx)] == 1)

        # each vertex one color
        for u in red_nodes:
            model.Add(sum(x[(u, c)] for c in range(H)) == 1)

        # adjacency constraints
        for u, v in red_edges:
            for c in range(H):
                model.Add(x[(u, c)] + x[(v, c)] <= w[c])

        # link w
        for c in range(H):
            model.Add(w[c] <= sum(x[(u, c)] for u in red_nodes))

        # symmetry breaking
        for c in range(1, H):
            model.Add(w[c-1] >= w[c])

        # objective
        model.Minimize(sum(w))

        # solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120.0
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []

        # ---- 6. Retrieve coloring on reduced graph ----
        col_red = {}
        for u in red_nodes:
            for c in range(H):
                if solver.Value(x[(u, c)]):
                    col_red[u] = c + 1
                    break

        # ---- 7. Map back to original graph ----
        result = [0]*n
        for v in range(n):
            root = v
            while dom[root] != root:
                root = dom[root]
            result[v] = col_red[root]

        # normalize colors to 1..k
        seen = sorted(set(result))
        map_col = {old: new for new, old in enumerate(seen, 1)}
        return [map_col[c] for c in result]