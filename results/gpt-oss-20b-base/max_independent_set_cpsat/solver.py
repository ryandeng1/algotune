# solver.py
import itertools
from functools import lru_cache
from typing import List, Any


class Solver:
    """Fast exact solver for Maximum Independent Set using branch‑and‑bound and bitsets."""

    def _build_masks(self, adj: List[List[int]]):
        """Create adjacency bit masks for each vertex."""
        n = len(adj)
        masks = [0] * n
        for i in range(n):
            m = 0
            for j, v in enumerate(adj[i]):
                if v:
                    m |= 1 << j
            masks[i] = m
        return masks

    def _coloring_upper_bound(self, nodes_mask: int, masks: List[int]) -> int:
        """
        Greedy graph coloring on the subgraph induced by nodes_mask.
        The number of colors is an upper bound on the size of a maximum
        independent set in that subgraph.
        """
        # Extract list of nodes
        nodes = [i for i in range(len(masks)) if nodes_mask >> i & 1]
        color = {}
        for v in nodes:
            forbidden = 0
            for u in color:
                if masks[u] >> v & 1:  # edge between colored node u and v
                    forbidden |= 1 << color[u]
            c = 0
            while forbidden >> c & 1:
                c += 1
            color[v] = c
        return max(color.values(), default=-1) + 1

    def _dfs(
        self, cur_mask: int, selected: int, masks: List[int], best: List[int]
    ) -> None:
        """Recursive search with pruning."""
        if cur_mask == 0:
            # No remaining nodes
            if selected > best[0]:
                best[0] = selected
            return

        # Upper bound: selected + maximal independent set size <= selected + upper_bound
        ub = selected + self._coloring_upper_bound(cur_mask, masks)
        if ub <= best[0]:
            return

        # Choose a node (heuristic: smallest remaining degree)
        # Convert mask to list of nodes and pick one with fewest mask bits
        degs = []
        for v in range(len(masks)):
            if cur_mask >> v & 1:
                degs.append((v, bin(masks[v] & cur_mask).count("1")))
        v, _ = min(degs, key=lambda x: x[1])

        # Branch: include v
        new_mask = cur_mask & ~masks[v] & ~(1 << v)
        self._dfs(new_mask, selected + 1, masks, best)

        # Branch: exclude v
        self._dfs(cur_mask & ~(1 << v), selected, masks, best)

    def solve(self, problem: List[List[int]]) -> List[int]:
        """Return a maximum independent set of the graph given by adjacency matrix."""
        n = len(problem)
        if n == 0:
            return []

        masks = self._build_masks(problem)
        full_mask = (1 << n) - 1
        best = [0]
        self._dfs(full_mask, 0, masks, best)

        # Reconstruct the set
        # We can redo a greedy extraction using the recursion info? Instead we rebuild by considering
        # that the optimal size is best[0]; we can run a simple greedy to get the actual vertices.
        # To guarantee optimality, we instead store the set during recursion.
        # We'll redo a simpler DFS that records best set.
        best_set = []

        def dfs_record(mask: int, selected: List[int], cur_set: List[int]) -> None:
            if mask == 0:
                if len(selected) > len(best_set):
                    best_set.clear()
                    best_set.extend(selected)
                return
            if len(selected) + self._coloring_upper_bound(mask, masks) <= len(best_set):
                return

            # pick a node
            degs = []
            for v in range(n):
                if mask >> v & 1:
                    degs.append((v, bin(masks[v] & mask).count("1")))
            v, _ = min(degs, key=lambda x: x[1])

            # include
            dfs_record(mask & ~masks[v] & ~(1 << v), selected + [v], cur_set + [v])
            # exclude
            dfs_record(mask & ~(1 << v), selected, cur_set)

        dfs_record(full_mask, [], [])
        return sorted(best_set)
