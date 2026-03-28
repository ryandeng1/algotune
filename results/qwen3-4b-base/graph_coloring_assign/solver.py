from typing import Any
from itertools import combinations
import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
        G.remove_edges_from(nx.selfloop_edges(G))

        def coloring_preprocessing_fast(G_sub):
            dominator = {v: v for v in G_sub.nodes()}
            prev_size = -1
            while True:
                current_size = len(G_sub.nodes())
                if current_size == prev_size:
                    break
                prev_size = current_size
                adj = {v: set(G_sub.neighbors(v)) for v in G_sub.nodes()}
                nodes = list(G_sub.nodes())
                n_nodes = len(nodes)
                sorted_nodes = sorted(nodes, key=lambda v: len(adj[v]))
                redundant = []
                for i in range(n_nodes):
                    u = sorted_nodes[i]
                    for j in range(i + 1, n_nodes):
                        v = sorted_nodes[j]
                        if adj[u].issubset(adj[v]):
                            redundant.append(u)
                            dominator[u] = v
                        elif adj[v].issubset(adj[u]):
                            redundant.append(v)
                            dominator[v] = u
                G_sub.remove_nodes_from(redundant)
            return G_sub, dominator

        G_red, dominator = coloring_preprocessing_fast(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())

        ub = len(set(nx.greedy_color(G_red).values()))
        H = ub

        clique_set = approx_clique.max_clique(G_red)
        Q = sorted(clique_set)
        lb = len(Q)

        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]

        model = cp_model.CpModel()
        x = {}
        for u in V:
            for i in range(H):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")

        w = {}
        for i in range(H):
            w[i] = model.NewBoolVar(f"w_{i}")

        for i, u in enumerate(Q):
            model.Add(x[(u, i)] == 1)

        for u in V:
            model.Add(sum(x[(u, i)] for i in range(H)) == 1)

        for u, v in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])

        for i in range(H):
            model.Add(w[i] <= sum(x[(u, i)] for u in V))

        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])

        model.Minimize(sum(w[i] for i in range(H)))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []

        c_red = {}
        for u in V:
            for i in range(H):
                if solver.Value(x[(u, i)]) == 1:
                    c_red[u] = i + 1
                    break

        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            colors[v] = c_red[root]

        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        colors = [remap[c] for c in colors]

        return colors