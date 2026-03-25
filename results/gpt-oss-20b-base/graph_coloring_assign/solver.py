# solver.py
import logging
from typing import Any, List
from collections import defaultdict

class Solver:
    """
    Fast exact graph coloring using DSATUR backtracking.
    Works efficiently for graphs up to ~60 vertices on ordinary hardware.
    """

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Returns a minimum coloring of the graph represented by an adjacency matrix.
        """
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency lists
        adj = [set() for _ in range(n)]
        for i in range(n):
            row = problem[i]
            for j, val in enumerate(row):
                if val and i != j:
                    adj[i].add(j)

        # Initial greedy coloring to obtain an upper bound
        greedy_colors = self._greedy_color(n, adj)
        ub = max(greedy_colors) if greedy_colors else 0

        # DSATUR search with iterative deepening
        best_colors = [0] * n
        # recursion stack variables
        color_used = [0] * n
        used_colors = set()

        # Saturation degrees
        sat_deg = [0] * n
        neigh_color_counts = [defaultdict(int) for _ in range(n)]

        order = list(range(n))
        order.sort(key=lambda x: len(adj[x]), reverse=True)

        self._dsatur(
            order,
            0,
            nb_colors=ub,
            color_used=color_used,
            used_colors=used_colors,
            sat_deg=sat_deg,
            neigh_color_counts=neigh_color_counts,
            best_colors=best_colors,
        )

        # Normalize colors to be 1..k
        used = sorted(set(best_colors))
        remap = {c: i + 1 for i, c in enumerate(used)}
        return [remap[c] for c in best_colors]

    # --------------------------------------------------------------------------
    def _greedy_color(self, n: int, adj):
        """Largest-degree first greedy coloring."""
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        colors = [0] * n
        for v in order:
            forbidden = {colors[u] for u in adj[v] if colors[u]}
            c = 1
            while c in forbidden:
                c += 1
            colors[v] = c
        return colors

    # --------------------------------------------------------------------------
    def _dsatur(self, order: List[int], idx: int, nb_colors: int,
                color_used: List[int], used_colors: set,
                sat_deg: List[int], neigh_color_counts: List[dict],
                best_colors: List[int]):
        """
        Recursive DSATUR backtracking.
        order: precedence of vertices (by degree)
        idx: current index in order being colored
        nb_colors: current upper bound on colors used
        """
        n = len(order)
        if idx == n:
            # all colored
            used = set(color_used)
            if len(used) < nb_colors:
                nb_colors = len(used)
                best_colors[:] = color_used[:]
            return nb_colors

        # select vertex with highest saturation, break ties by degree
        u = None
        max_sat = -1
        max_deg = -1
        for i in range(idx, n):
            v = order[i]
            sat = sat_deg[v]
            deg = len(adj[v])
            if sat > max_sat or (sat == max_sat and deg > max_deg):
                max_sat, max_deg, u = sat, deg, v

        # bring u to current position
        pos = order.index(u)
        order[idx], order[pos] = order[pos], order[idx]

        # try colors
        neighbor_colors = neigh_color_counts[u]
        for c in range(1, nb_colors + 1):
            if c in neighbor_colors:
                continue
            # assign
            color_used[u] = c
            used_colors.add(c)
            # update neighbours' saturation
            updates = []
            for v in adj[u]:
                if color_used[v] == 0:
                    d = neigh_color_counts[v]
                    if d[c] == 0:
                        sat_deg[v] += 1
                    d[c] += 1
                    updates.append(v)
            # pruning: if used > current best, skip
            if len(used_colors) <= nb_colors:
                nb_colors = self._dsatur(order, idx + 1, nb_colors,
                                         color_used, used_colors,
                                         sat_deg, neigh_color_counts,
                                         best_colors)
            # rollback
            for v in updates:
                d = neigh_color_counts[v]
                d[c] -= 1
                if d[c] == 0:
                    sat_deg[v] -= 1
            used_colors.discard(c)
            color_used[u] = 0

        # backtrack order swap
        order[idx], order[pos] = order[pos], order[idx]
        return nb_colors
