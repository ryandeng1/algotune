from typing import List
import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model
from itertools import combinations

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve graph coloring using CP‑SAT with clique seeding and symmetry breaking.
        """
        n = len(problem)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)

        # Pre‑processing: remove dominated vertices
        def reduce_graph(G_sub):
            dominator = {v: v for v in G_sub.nodes()}
            while True:
                prev = set(G_sub.nodes())
                adj = {v: set(G_sub.neighbors(v)) for v in G_sub.nodes()}
                redundant = []
                for u, v in combinations(G_sub.nodes(), 2):
                    if adj[u] <= adj[v]:
                        redundant.append(u)
                        dominator[u] = v
                    elif adj[v] <= adj[u]:
                        redundant.append(v)
                        dominator[v] = u
                if not redundant:
                    break
                G_sub.remove_nodes_from(redundant)
            return G_sub, dominator

        G_red, dominator = reduce_graph(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())

        # Upper bound from greedy coloring
        ub = len(set(nx.greedy_color(G_red).values()))
        # Lower bound from maximum clique
        cliq = approx_clique.max_clique(G_red)
        lb = len(cliq)

        # If bounds coincide, the greedy solution is optimal
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]

        # ---------------- CP‑SAT model ----------------
        model = cp_model.CpModel()
        H = ub  # upper bound on number of colors

        # Variables: x[u,i] = 1 if vertex u gets color i
        x = {(u, i): model.NewBoolVar(f"x_{u}_{i}") for u in V for i in range(H)}
        # Variables: w[i] = 1 if color i is used
        w = [model.NewBoolVar(f"w_{i}") for i in range(H)]

        # Clique nodes must use distinct colors 0..|Q|-1
        for idx, u in enumerate(cliq):
            model.Add(x[(u, idx)] == 1)

        # Each vertex exactly one color
        for u in V:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)

        # Adjacent vertices can't share a color
        for u, v in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])

        # Link w and x
        for i in range(H):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))

        # Symmetry breaking: colors used in decreasing order
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])

        # Objective: minimize number of colors used
        model.Minimize(sum(w))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 25.0  # give some time, but keep reasonable
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return []

        # Extract colors for reduced graph
        c_red = {}
        for u in V:
            for i in range(H):
                if solver.Value(x[(u, i)]) == 1:
                    c_red[u] = i + 1  # colors 1..H
                    break

        # Map dominator roots to colors
        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = c_red[root]

        # Remap colors to consecutive 1..k
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors = [remap[c] for c in colors]
        return colors