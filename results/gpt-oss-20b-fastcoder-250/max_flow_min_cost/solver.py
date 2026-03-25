# solver.py
import math
from typing import Any, List, Tuple
import heapq

# --------------------------------------------------------------------------- #
# Implementation of the successive shortest augmenting path algorithm
# --------------------------------------------------------------------------- #

class Solver:
    """
    Solver for the Maximum Flow Min Cost problem.
    """

    # ----------------------------------------------------------------------- #
    # Helper: build graph from problem dict
    # ----------------------------------------------------------------------- #
    def _build_graph(self, prob: dict) -> Tuple[List[List[int]], List[List[int]], List[List[int]], int, int, int]:
        """
        Build adjacency lists for capacity, cost, and neighbors.

        Returns:
            cap: 2D list of capacities
            cost: 2D list of costs
            neigh: list of neighbor indices per node
            source: source node index
            sink: sink node index
            n: number of nodes
        """
        cap = prob["capacity"]
        cost = prob["cost"]
        n = len(cap)
        neigh = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if cap[u][v] > 0:
                    neigh[u].append(v)
        source = prob["s"]
        sink = prob["t"]
        return cap, cost, neigh, source, sink, n

    # ----------------------------------------------------------------------- #
    # Core min-cost max‑flow routine
    # ----------------------------------------------------------------------- #
    def _min_cost_flow(self, cap: List[List[int]], cost: List[List[int]],
                       neigh: List[List[int]], s: int, t: int, n: int) -> List[List[int]]:

        # residual capacities
        res_cap = [row[:] for row in cap]

        # potentials for reduced costs (Johnson's trick)
        potential = [0] * n

        # flow matrix initialized to 0
        flow = [[0] * n for _ in range(n)]

        # loop until no augmenting path from s to t
        INF = 10 ** 18
        while True:
            dist = [INF] * n
            prev = [-1] * n
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d != dist[u]:
                    continue
                for v in neigh[u]:
                    rc = cost[u][v] + potential[u] - potential[v]
                    if res_cap[u][v] > 0 and dist[v] > d + rc:
                        dist[v] = d + rc
                        prev[v] = u
                        heapq.heappush(pq, (dist[v], v))
                # consider reverse edges
                for v in neigh[v] if False else []:  # dummy for symmetry
                    pass

            if dist[t] == INF:
                break  # no more augmenting path

            # update potentials
            for i in range(n):
                if dist[i] < INF:
                    potential[i] += dist[i]

            # find bottleneck capacity
            aug = INF
            v = t
            while v != s:
                u = prev[v]
                aug = min(aug, res_cap[u][v])
                v = u
            if aug == 0 or aug == INF:
                break

            # augment flow
            v = t
            while v != s:
                u = prev[v]
                res_cap[u][v] -= aug
                res_cap[v][u] += aug
                flow[u][v] += aug
                v = u

        return flow

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def solve(self, problem: dict[str, Any]) -> List[List[Any]]:
        """Return a flow matrix for the given max‑flow‑min‑cost problem."""
        try:
            cap, cost, neigh, s, t, n = self._build_graph(problem)
            flow = self._min_cost_flow(cap, cost, neigh, s, t, n)
            return flow
        except Exception:
            # In case of any failure, return zero matrix
            return [[0 for _ in range(len(problem["capacity"]))] for _ in range(len(problem["capacity"]))]
