# solver.py
from typing import List

import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model


class Solver:
    """Fast CP‑SAT graph‑coloring solver."""

    @staticmethod
    def _remap_colors(colors: List[int]) -> List[int]:
        """Remap sparse colour indices to 1..k."""
        uniq = sorted(set(colors))
        remap = {c: i + 1 for i, c in enumerate(uniq)}
        return [remap[c] for c in colors]

    @staticmethod
    def _make_graph(adj: List[List[int]]) -> nx.Graph:
        """Create a NetworkX graph from an adjacency matrix."""
        n = len(adj)
        G = nx.from_numpy_array(
            np.array(adj, dtype=np.bool_), create_using=nx.Graph()
        )
        # if the input has self‑loops, NetworkX removes them automatically
        return G

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Colour a graph using a classical CP‑SAT formulation with clique seeding
        and symmetry‑breaking.

        :param problem: 2‑d list adjacency matrix
        :return: 1‑based colour assignment or [] if infeasible
        """
        # ------------------------------------------------------------------
        # 1. Build a lightweight graph (avoid the double loop of the original)
        # ------------------------------------------------------------------
        import numpy as np

        G = self._make_graph(problem)
        n = len(problem)

        # ------------------------------------------------------------------
        # 2. Pre‑processing – remove redundant vertices (dominance pruning)
        # ------------------------------------------------------------------
        def prune(G_orig: nx.Graph):
            # Map vertex -> representative (itself until changed)
            rep = {v: v for v in G_orig}
            G_adj = {v: set(G_orig.neighbors(v)) for v in G_orig}
            changed = True
            while changed:
                changed = False
                nodes = list(G_adj)
                for i, u in enumerate(nodes):
                    for v in nodes[i + 1 :]:
                        if G_adj[u] <= G_adj[v]:
                            G_adj[u] = set()  # mark for deletion
                            rep[u] = rep[v]
                            changed = True
                        elif G_adj[v] <= G_adj[u]:
                            G_adj[v] = set()
                            rep[v] = rep[u]
                            changed = True
                # delete marked vertices
                to_del = [v for v, neigh in G_adj.items() if not neigh]
                G_adj = {v: neigh for v, neigh in G_adj.items() if neigh}
            G_red = nx.Graph.from_edgelist(
                [(u, v) for u in G_adj for v in G_adj[u] if u < v]
            )
            return G_red, rep

        G_red, rep = prune(G.copy())

        # ------------------------------------------------------------------
        # 3. Upper/lower bounds from greedy colour / max clique
        # ------------------------------------------------------------------
        ub = len(set(nx.greedy_color(G_red).values()))
        best_clique = approx_clique.max_clique(G_red)
        lb = len(best_clique)

        if lb == ub:  # trivial optimal solution
            greedy = nx.greedy_color(G, strategy="largest_first")
            return [greedy[i] + 1 for i in range(n)]

        # ------------------------------------------------------------------
        # 4. CP‑SAT model
        # ------------------------------------------------------------------
        h = ub  # upper bound on colours
        model = cp_model.CpModel()

        # colour variables
        x = {
            (v, c): model.NewBoolVar(f"x_{v}_{c}") for v in G_red for c in range(h)
        }

        # colour used variables
        w = {c: model.NewBoolVar(f"w_{c}") for c in range(h)}

        # enforce initial clique colours
        for i, v in enumerate(sorted(best_clique)):
            model.Add(x[(v, i)] == 1)

        # each vertex gets exactly one colour
        for v in G_red:
            model.Add(sum(x[(v, c)] for c in range(h)) == 1)

        # adjacent vertices cannot share a colour
        for u, v in G_red.edges():
            for c in range(h):
                model.Add(x[(u, c)] + x[(v, c)] <= w[c])

        # colour used only if any vertex takes it
        for c in range(h):
            model.Add(w[c] <= sum(x[(v, c)] for v in G_red))

        # colour usage is non‑increasing
        for c in range(1, h):
            model.Add(w[c - 1] >= w[c])

        model.Minimize(sum(w[c] for c in range(h)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # optional time limit
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return []

        # ------------------------------------------------------------------
        # 5. Build solution for the original graph
        # ------------------------------------------------------------------
        colour_red = {
            v: next(
                c + 1
                for c in range(h)
                if solver.Value(x[(v, c)]) == 1
            )
            for v in G_red
        }

        colours = [0] * n
        for v in range(n):
            root = rep.get(v, v)
            while rep.get(root, root) != root:
                root = rep[root]
            colours[v] = colour_red[root]

        return self._remap_colors(colours)