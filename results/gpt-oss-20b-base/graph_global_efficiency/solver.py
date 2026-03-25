# solver.py
import math
import heapq
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
        """
        Compute the global efficiency of an undirected graph given as an adjacency list.

        Global efficiency = (1/(N*(N-1))) * sum_{u!=v} (1 / d(u,v))
        where d(u,v) is the shortest path length (inf if disconnected).

        Parameters
        ----------
        problem : dict
            Must contain key "adjacency_list" mapping to a list of lists of integers.

        Returns
        -------
        dict
            {"global_efficiency": float}
        """
        adj = problem.get("adjacency_list", [])
        n = len(adj)

        # Edge cases: 0 or 1 node => efficiency 0
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Precompute shortest path sums via Dijkstra from each node.
        # Use a heap queue for efficiency on sparse graphs.
        total_inv = 0.0

        for src in range(n):
            # distances initialized to inf
            dist = [math.inf] * n
            dist[src] = 0
            heap = [(0, src)]

            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]:
                    continue
                for v in adj[u]:
                    nd = d + 1.0  # unweighted graph: each edge weight = 1
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))

            # Sum inverse distances (exclude self)
            for v in range(n):
                if v == src:
                    continue
                d = dist[v]
                if d < math.inf:
                    total_inv += 1.0 / d

        # Normalize by number of ordered pairs N*(N-1)
        efficiency = total_inv / (n * (n - 1))
        return {"global_efficiency": float(efficiency)}
