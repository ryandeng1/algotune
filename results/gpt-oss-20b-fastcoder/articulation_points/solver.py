# solver.py

from typing import Any, Dict, List

class Solver:
    """
    Implementation of articulation point detection via an iterative Tarjan's algorithm.
    This version avoids heavy dependencies on networkx and therefore achieves faster
    runtimes, especially on large sparse graphs.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Parameters
        ----------
        problem : dict
            Must contain:
                'num_nodes' : int
                'edges'     : list[tuple[int, int]]

        Returns
        -------
        dict
            A mapping with key 'articulation_points' containing a sorted list
            of articulation point indices.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adjacency: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        # Tarjan's algorithm variables
        disc = [-1] * n          # discovery times
        low = [0] * n            # low-link values
        parent = [-1] * n
        ap = [False] * n
        time = 0

        stack = []

        for start in range(n):
            if disc[start] != -1:
                continue  # already visited

            # Start DFS from `start`
            stack.append((start, 0, iter(adjacency[start])))
            disc[start] = low[start] = time
            time += 1

            while stack:
                node, child_idx, neighbors = stack[-1]
                try:
                    nbr = next(neighbors)
                except StopIteration:
                    # Finished exploring all neighbors of `node`
                    stack.pop()
                    if parent[node] != -1:
                        # Update low of parent
                        if low[node] < low[parent[node]]:
                            low[parent[node]] = low[node]
                        # Articulation point check for non-root
                        if low[node] >= disc[parent[node]]:
                            ap[parent[node]] = True
                    continue

                if disc[nbr] == -1:
                    # Tree edge
                    parent[nbr] = node
                    disc[nbr] = low[nbr] = time
                    time += 1
                    stack.append((nbr, 0, iter(adjacency[nbr])))
                elif nbr != parent[node]:
                    # Back edge
                    if disc[nbr] < low[node]:
                        low[node] = disc[nbr]

            # After finishing a connected component, check root condition
            # Root is the first element of the component
            if parent[start] == -1:
                child_count = 0
                for nb in adjacency[start]:
                    if parent[nb] == start:
                        child_count += 1
                if child_count > 1:
                    ap[start] = True

        # Gather and sort articulation points
        articulation_points = [i for i, val in enumerate(ap) if val]
        articulation_points.sort()
        return {"articulation_points": articulation_points}