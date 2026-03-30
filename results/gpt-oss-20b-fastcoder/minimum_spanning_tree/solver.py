#!/usr/bin/env python3
# solver.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import heapq

# --------------------------------------------------------------------------- #
#  Fast Minimum Spanning Tree (Prim's algorithm) implementation.
#  No external dependencies except the standard library.
# --------------------------------------------------------------------------- #


class Solver:
    """Solver that computes a Minimum Spanning Tree (MST) for an undirected
    weighted graph. The graph is defined in the input problem dictionary:
      - problem['num_nodes']: int, number of nodes (0 .. n-1)
      - problem['edges']   : List[Tuple[int, int, float]],
        each tuple (u, v, weight) represents an undirected edge.
    The output dictionary contains a single key 'mst_edges', mapping to a
    list of edges, each represented as [u, v, weight] with u <= v.
    The list is sorted lexicographically by (u, v).
    """

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """Compute the MST and return it in the expected format."""
        num_nodes: int = problem["num_nodes"]
        edges: List[Tuple[int, int, float]] = problem["edges"]

        # Build adjacency list: node -> list of (neighbor, weight)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Prim's algorithm – adjacency heap approach
        visited = [False] * num_nodes
        min_heap: List[Tuple[float, int, int]] = []  # (weight, u, v)
        mst: List[Tuple[int, int, float]] = []

        # Start from node 0 (any node works because the graph is connected)
        visited[0] = True
        for v, w in adj[0]:
            heapq.heappush(min_heap, (w, 0, v))

        count = 1  # number of nodes already in MST
        while min_heap and count < num_nodes:
            w, u, v = heapq.heappop(min_heap)
            if visited[v]:
                continue  # skip already-in‑MST nodes
            # Add this edge to MST
            if u > v:
                u, v = v, u
            mst.append((u, v, w))
            visited[v] = True
            count += 1
            # Push all edges from the newly visited node
            for nb, nb_w in adj[v]:
                if not visited[nb]:
                    heapq.heappush(min_heap, (nb_w, v, nb))

        # Sort the MST edges lexicographically by (u, v)
        mst.sort(key=lambda e: (e[0], e[1]))

        # Convert to list of lists for the required return type
        mst_edges = [[u, v, w] for u, v, w in mst]
        return {"mst_edges": mst_edges}