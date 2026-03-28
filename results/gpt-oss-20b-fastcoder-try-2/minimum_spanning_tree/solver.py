from typing import Any, List, Tuple, Dict
import heapq

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph using
        Prim's algorithm with a binary heap. The input graph is given as a list of
        edges `(u, v, w)` where `u` and `v` are zero‑based node indices and `w` is the
        positive weight.

        The output is a list of edges in the MST sorted by `(u, v)` pairs. Edges are
        represented as `[u, v, w]` with the smaller endpoint first.
        """
        num_nodes: int = problem['num_nodes']
        edges: List[Tuple[int, int, float]] = problem['edges']

        # Build adjacency list
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Prim's algorithm
        visited = [False] * num_nodes
        mst = []
        # Start from node 0
        visited[0] = True
        heap: List[Tuple[float, int, int]] = []  # (weight, from, to)

        for to, w in adj[0]:
            heapq.heappush(heap, (w, 0, to))

        while heap and len(mst) < num_nodes - 1:
            w, frm, to = heapq.heappop(heap)
            if visited[to]:
                continue
            visited[to] = True
            # store edge with smaller endpoint first
            if frm > to:
                frm, to = to, frm
            mst.append([frm, to, w])
            for nxt, nw in adj[to]:
                if not visited[nxt]:
                    heapq.heappush(heap, (nw, to, nxt))

        # Sort edges by (u, v)
        mst.sort(key=lambda e: (e[0], e[1]))
        return {'mst_edges': mst}