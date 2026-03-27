from typing import List, Set


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the maximum clique problem using a fast recursive Bron–Kerbosch
        algorithm with pivoting and pruning. The graph is represented by an
        adjacency matrix. The returned list contains the indices of the
        vertices in one maximum clique.
        """
        n = len(problem)
        # Build adjacency sets for quick look‑ups
        adjacency: List[Set[int]] = [set(j for j, val in enumerate(row) if val) for row in problem]
        max_clique: List[int] = []

        def bronk(R: Set[int], P: Set[int], X: Set[int]) -> None:
            nonlocal max_clique
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = sorted(R)
                return
            # Pivot selection: choose a vertex u from P ∪ X that maximizes |P∩N(u)|
            u = None
            max_deg = -1
            for v in P.union(X):
                deg = len(P.intersection(adjacency[v]))
                if deg > max_deg:
                    max_deg = deg
                    u = v
            # Explore vertices not in neighbors of pivot
            for v in P - adjacency[u]:
                bronk(R | {v}, P & adjacency[v], X & adjacency[v])
                P.remove(v)
                X.add(v)

        bronk(set(), set(range(n)), set())
        return max_clique