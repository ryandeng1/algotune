# solver.py
# Optimised CP‑SAT graph‑colouring solver
# -------------------------------------------------
# The code removes unnecessary `else: pass` statements, replaces
# nested loops that iterated over pairs of vertices with a neighbour
# dominance test that uses Python sets, and uses a one‑liner
# conversion from the CP‑model solution to the final colour list.
#
# The heavy CP‑model part is kept only when a gap between the lower
# bound (size of a maximum clique) and the upper bound (greedy
# colouring bound) is detected.  For most practical instances the
# greedy colouring resolves the problem instantly.

from __future__ import annotations
import sys
from itertools import combinations
from typing import List

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model


class Solver:
    """CP‑SAT based graph‑colouring solver with lightweight preprocessing."""

    # ----------------------------------------------------------------------
    #  Core solver
    # ----------------------------------------------------------------------
    def solve(self, problem: List[List[int]]) -> List[int]:
        """Return a list of colours (1‑based) for the vertices of the graph."""
        n = len(problem)

        # --------------------------------------------------------------
        #  Build the graph
        # --------------------------------------------------------------
        G = nx.from_numpy_array(np.array(problem, dtype=bool))

        # --------------------------------------------------------------
        #  Pre‑processing: remove dangling vertices that are dominated
        #  by other vertices (they never require a distinct colour).
        # --------------------------------------------------------------
        def reduce_graph(G_sub: nx.Graph):
            # Initialise each vertex as its own dominator
            dominator = {v: v for v in G_sub.nodes()}
            prev_len = -1
            while len(G_sub) != prev_len:
                prev_len = len(G_sub)
                neigh = {v: set(G_sub.neighbors(v)) for v in G_sub}
                disc = set()
                for u, v in combinations(G_sub, 2):
                    if neigh[u] <= neigh[v]:
                        disc.add(u)
                        dominator[u] = v
                    elif neigh[v] <= neigh[u]:
                        disc.add(v)
                        dominator[v] = u
                G_sub.remove_nodes_from(disc)
            return G_sub, dominator

        G_red, dominator = reduce_graph(G.copy())

        # --------------------------------------------------------------
        #  Bounds
        # --------------------------------------------------------------
        ub = len(set(nx.greedy_color(G_red, strategy="largest_first").values()))
        clique_vertices = sorted(nx.algorithms.approximation.clique.max_clique(G_red))
        lb = len(clique_vertices)

        # If the bounds coincide, simply return the greedy colouring.
        if lb == ub:
            greedy = nx.greedy_color(G, strategy="largest_first")
            # +1 to make 1‑based
            return [greedy[i] + 1 for i in range(n)]

        # --------------------------------------------------------------
        #  CP‑SAT model
        # --------------------------------------------------------------
        H = ub  # maximum number of colours
        V = list(G_red.nodes())
        E = list(G_red.edges())

        model = cp_model.CpModel()
        # binary variables: x[v,i] = 1 iff vertex v uses colour i
        x = {(v, i): model.NewBoolVar(f"x_{v}_{i}") for v in V for i in range(H)}
        # colour usage variables
        w = [model.NewBoolVar(f"w_{i}") for i in range(H)]

        # Force vertices of the maximum clique to use the first |clique| colours
        for i, v in enumerate(clique_vertices):
            model.Add(x[(v, i)] == 1)

        # Each vertex uses exactly one colour
        for v in V:
            model.Add(sum(x[(v, i)] for i in range(H)) == 1)

        # No two adjacent vertices share a colour unless that colour is used
        for (u, v) in E:
            for i in range(H):
                model.Add(x[(u, i)] + x[(v, i)] <= w[i])

        # Colour usage is implied by at least one vertex using it
        for i in range(H):
            model.Add(w[i] <= sum(x[(v, i)] for v in V))

        # Colours are used in non‑increasing order to break symmetry
        for i in range(1, H):
            model.Add(w[i - 1] >= w[i])

        # Objective: minimise number of colours used
        model.Minimize(sum(w))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 20.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # --------------------------------------------------------------
        #  Reconstruct colours for all vertices
        # --------------------------------------------------------------
        # Colour of a vertex is the colour of its dominator
        colors = [0] * n
        for v in range(n):
            root = v
            while dominator[root] != root:
                root = dominator[root]
            for i in range(H):
                if solver.Value(x[(root, i)]) == 1:
                    colors[v] = i + 1
                    break

        # Remap colours to use consecutive numbers starting from 1
        used = sorted(set(colors))
        remap = {c: i + 1 for i, c in enumerate(used)}
        return [remap[c] for c in colors]


# ----------------------------------------------------------------------
#  Quick sanity test (removed from the final module in real use)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
    ]
    print(Solver().solve(sample))