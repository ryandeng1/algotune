from typing import List, Dict
from itertools import combinations
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the graph coloring problem with a CP-SAT model.
        The input is an adjacency matrix (list of lists) where
        problem[i][j] == 1 means that vertices i and j are adjacent.
        Returns a list of colors (1‑based) for each vertex.
        """

        n = len(problem)

        # ---------- Build adjacency representation ----------
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    neighbors[i].add(j)
        for i in range(n):
            neighbors[i].discard(i)          # remove self‑loops, if any

        # ---------- Dominator preprocessing ----------
        def dominator_preprocess(nodes: List[int]) -> List[int]:
            # nodes is a list of vertices that still exist
            dominator: Dict[int, int] = {v: v for v in nodes}
            changed = True
            while changed:
                changed = False
                adj = {v: neighbors[v].intersection(nodes) for v in nodes}
                to_remove = []
                for u, v in combinations(nodes, 2):
                    if adj[u].issubset(adj[v]):
                        to_remove.append(u)
                        dominator[u] = v
                    elif adj[v].issubset(adj[u]):
                        to_remove.append(v)
                        dominator[v] = u
                if to_remove:
                    changed = True
                    for r in to_remove:
                        nodes.remove(r)
            return nodes, dominator

        nodes_red, dominator = dominator_preprocess(list(range(n)))

        # ---------- Upper bound via simple greedy ----------
        # Greedy uses smallest possible color each time
        color_greedy = [0] * n
        max_color = 0
        for u in range(n):
            forbidden = {color_greedy[v] for v in neighbors[u] if color_greedy[v]}
            c = 1
            while c in forbidden:
                c += 1
            color_greedy[u] = c
            max_color = max(max_color, c)
        ub = max_color

        # ---------- Heuristic largest clique ----------
        # Simple greedy clique enumeration
        def max_clique_greedy():
            order = sorted(range(n), key=lambda v: -len(neighbors[v]))
            clique = []
            used = set()
            for v in order:
                if all(v in neighbors[u] for u in clique):
                    clique.append(v)
            return clique

        clique = max_clique_greedy()
        lb = len(clique)

        # If clique size equals the greedy upper bound, return the greedy solution
        if lb == ub:
            return [color_greedy[v] for v in range(n)]

        H = ub  # number of color slots

        # ---------- CP‑SAT model ----------
        model = cp_model.CpModel()

        # Variables: x[u,i] = 1 if vertex u uses color i+1
        x: Dict[tuple, cp_model.BoolVar] = {}
        for u in nodes_red:
            for i in range(H):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")

        # Variables: w[i] indicate whether color i+1 is used
        w = [model.NewBoolVar(f"w_{i}") for i in range(H)]

        # Seed clique with distinct colors
        for idx, v in enumerate(clique):
            if v in nodes_red:  # the clique node might have been removed
                model.Add(x[(v, idx)] == 1)

        # (1) Exactly one color per vertex
        for u in nodes_red:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)

        # (2) Adjacent vertices cannot share a color unless w[i] == 1
        for u in nodes_red:
            for v in neighbors[u]:
                if u < v and v in nodes_red:
                    for i in range(H):
                        model.Add(x[(u, i)] + x[(v, i)] <= w[i])

        # (3) Link w[i] to assignments
        for i in range(H):
            model.Add(sum(x[(u, i)] for u in nodes_red) >= w[i])

        # (4) Symmetry breaking: w[0] >= w[1] >= ...
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])

        # Objective: minimize number of colors used
        model.Minimize(sum(w))

        # ---------- Solve ----------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL:
            return []

        # ---------- Extract color assignment ----------
        color_red: Dict[int, int] = {}
        for u in nodes_red:
            for i in range(H):
                if solver.Value(x[(u, i)]):
                    color_red[u] = i + 1
                    break

        # ---------- Map back through dominator ----------
        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = color_red[root]

        # ---------- Normalize colors to consecutive 1..k ----------
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors = [remap[c] for c in colors]

        return colors