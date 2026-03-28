import numpy as np

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-06

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        adj_list = problem['adjacency_list']
        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build transition matrix as a dense array (fast for moderate n)
        transition = np.zeros((n, n), dtype=np.float64)
        out_deg = [len(neigh) for neigh in adj_list]
        for u, neigh in enumerate(adj_list):
            if out_deg[u] == 0:
                # dangling node: distribute uniformly
                transition[u, :] = 1.0 / n
            else:
                weight = 1.0 / out_deg[u]
                transition[u, neigh] = weight

        # Precompute teleport vector
        teleport = np.full(n, 1.0 / n, dtype=np.float64)

        # Power iteration
        pr = np.full(n, 1.0 / n, dtype=np.float64)
        for _ in range(self.max_iter):
            new_pr = self.alpha * transition.T @ pr + (1.0 - self.alpha) * teleport
            if np.linalg.norm(new_pr - pr, 1) < self.tol:
                pr = new_pr
                break
            pr = new_pr

        return {"pagerank_scores": pr.tolist()}