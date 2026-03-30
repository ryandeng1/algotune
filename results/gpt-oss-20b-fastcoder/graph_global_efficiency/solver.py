# solver.py
"""
Optimised implementation of the global efficiency computation.
The original code used NetworkX, which is heavy for large graphs.
Here we compute the efficiency directly with a BFS from each node,
avoiding the overhead of NetworkX and customising the algorithm
for speed on typical sparse graphs.

Implementation notes
--------------------
* The graph is read from `problem['adjacency_list']`.  It is assumed to be
  an adjacency list of an *undirected* simple graph.  The nodes are
  labelled 0 … n‑1.
* For each node `s` we run a breadth‑first search (BFS) to obtain the
  shortest‑path distances to every other reachable node.  Because the
  graph is unweighted, BFS gives the exact shortest distance in terms
  of hop‑count.
* While exploring a node we immediately add the contribution
  `1 / dist` to a running total; this avoids storing the full distance
  matrix and keeps memory usage low.
* The total number of ordered pairs considered is `n*(n-1)`.  The global
  efficiency is the average of the inverses of the distances over all
  *ordered* pairs (NodeA → NodeB and NodeB → NodeA are counted
  separately).  For disconnected pairs we add `0`.
* Complexity: O(n·(n+m)) where `m` is the number of edges.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List


class Solver:
    """
    Compute the global efficiency of an undirected simple graph.

    The method ``solve`` accepts a dictionary
        {"adjacency_list": List[List[int]]}

    and returns a dictionary
        {"global_efficiency": float}
    """

    @staticmethod
    def _bfs_sum_inverse(
        src: int,
        adj: List[List[int]],
        n: int,
        visited: List[bool],
        dist: List[int],
    ) -> float:
        """
        Run a BFS from ``src`` and return the sum of 1/distance
        for all vertices reachable from ``src``.
        Vertices that are not reachable contribute 0.
        """
        visited[src] = True
        dist[src] = 0
        queue: deque[int] = deque([src])
        total: float = 0.0

        while queue:
            u = queue.popleft()
            d_u = dist[u]
            # skip adding self‑pair (d=0)
            if u != src:
                total += 1.0 / d_u

            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = d_u + 1
                    queue.append(v)

        return total

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
        adj = problem.get("adjacency_list", [])
        n = len(adj)

        if n <= 1:
            return {"global_efficiency": 0.0}

        # allocate reusable buffers – they are reused for every source
        visited = [False] * n
        dist = [0] * n

        # accumulate the sum of 1/d over all ordered pairs
        sum_inv_dist: float = 0.0

        for src in range(n):
            # reset visited/dist for next source
            for i in range(n):
                visited[i] = False
                dist[i] = 0

            sum_inv_dist += self._bfs_sum_inverse(src, adj, n, visited, dist)

        # total number of ordered pairs (i, j), i != j
        total_pairs = n * (n - 1)

        efficiency = sum_inv_dist / total_pairs
        return {"global_efficiency": float(efficiency)}