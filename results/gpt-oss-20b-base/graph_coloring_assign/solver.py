from typing import List
import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves graph coloring with CP‑SAT. The implementation focuses on speed:
        * minimal Python overhead
        * efficient graph reduction
        * early exit when greedy coloring is optimal
        """
        n = len(problem)
        G = nx.from_numpy_array(
            np.array(problem, dtype=bool), create_using=nx.Graph
        )
        # Clean self‑loops (should not exist but be safe)
        G.remove_edges_from(nx.selfloop_edges(G))

        # ----------------------------------------------------
        # 1. Dominance reduction (remove redundant vertices)
        # ----------------------------------------------------
        def reduce_graph(Gt):
            dom = {v: v for v in Gt.nodes()}
            prev = -1
            while len(Gt) != prev:
                prev = len(Gt)
                neigh = {v: set(Gt.neighbors(v)) for v in Gt}
                to_remove = []
                nodes = list(Gt)
                for i in range(len(nodes)):
                    u = nodes[i]
                    if to_remove and u in to_remove: continue
                    for v in nodes[i+1:]:
                        if v in to_remove: continue
                        if neigh[u] <= neigh[v]:
                            to_remove.append(u)
                            dom[u] = v
                            break
                        if neigh[v] <= neigh[u]:
                            to_remove.append(v)
                            dom[v] = u
                Gt.remove_nodes_from(to_remove)
            return Gt, dom

        G_red, dominator = reduce_graph(G.copy())
        V = list(G_red.nodes())
        E = list(G_red.edges())

        ub = nx.greedy_color(G_red).max() + 1  # upper bound
        lb = len(approx_clique.max_clique(G_red))  # lower bound

        # If bounds coincide, greedy coloring is optimal
        if ub == lb:
            greedy = nx.greedy_color(G, strategy='largest_first')
            return [greedy[i] + 1 for i in range(n)]

        H = ub  # number of colors at most

        # ----------------------------------------------------
        # 2. CP‑SAT model
        # ----------------------------------------------------
        model = cp_model.CpModel()
        # boolean vars: x[u,i] = vertex u colored i (0‑based)
        x = {}
        for u in V:
            for i in range(H):
                x[u, i] = model.NewBoolVar(f'x_{u}_{i}')
        # bool var: w[i] = colors[i] used
        w = [model.NewBoolVar(f'w_{i}') for i in range(H)]

        # enforce lower bound colors (clique)
        clique = approx_clique.max_clique(G_red)
        for i, u in enumerate(clique):
            model.Add(x[u, i] == 1)

        # each vertex one color
        for u in V:
            model.Add(sum(x[u, i] for i in range(H)) == 1)

        # adjacency constraints
        for u, v in E:
            for i in range(H):
                model.Add(x[u, i] + x[v, i] <= w[i])

        # w[i] <= sum x[...,i]
        for i in range(H):
            model.Add(w[i] <= sum(x[u, i] for u in V))

        # non‑increasing w
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])

        model.Minimize(sum(w))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8  # parallelism
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL:
            return []

        # ----------------------------------------------------
        # 3. Recover colors (including reduced vertices)
        # ----------------------------------------------------
        c_red = {}
        for u in V:
            for i in range(H):
                if solver.Value(x[u, i]) == 1:
                    c_red[u] = i + 1
                    break

        # map back to original vertices
        colors = [0] * n
        for v in range(n):
            # follow dominator chain to representative
            while dominator[v] != v:
                v = dominator[v]
            colors[v] = c_red[v]

        # remap colors to be consecutive starting at 1
        used = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used, start=1)}
        return [remap[c] for c in colors]