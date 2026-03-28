import math
from typing import Dict, List

class Solver:
    def __init__(self, alpha: float = 0.85, max_iter: int = 200, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        adj_list = problem['adjacency_list']
        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Preprocess: compute outdegree, list of outbound neighbours
        out_deg = [len(neigh) for neigh in adj_list]
        dangling = [i for i, deg in enumerate(out_deg) if deg == 0]

        # Normalized transition matrix rows as lists of (dest, weight)
        transitions = []
        for u, neigh in enumerate(adj_list):
            if out_deg[u] > 0:
                weight = 1.0 / out_deg[u]
                transitions.append([(v, weight) for v in neigh])
            else:
                transitions.append([])

        # Initialize PR vector
        pr = [1.0 / n] * n
        teleport = (1.0 - self.alpha) / n

        for _ in range(self.max_iter):
            new_pr = [teleport] * n
            # Handle dangling nodes
            dangling_sum = sum(pr[u] for u in dangling)
            if dangling_sum:
                add = self.alpha * dangling_sum / n
                new_pr = [x + add for x in new_pr]
            # Contributions from non-dangling nodes
            for u, neighbors in enumerate(transitions):
                if neighbors:
                    contrib = pr[u] * self.alpha
                    for v, w in neighbors:
                        new_pr[v] += contrib * w
            # Check convergence
            diff = sum(abs(new_pr[i] - pr[i]) for i in range(n))
            if diff < self.tol:
                pr = new_pr
                break
            pr = new_pr

        return {"pagerank_scores": [float(v) for v in pr]}