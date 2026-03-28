import numpy as np

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-06

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        adj = problem['adjacency_list']
        n = len(adj)
        if n == 0:
            return {'pagerank_scores': []}
        if n == 1:
            return {'pagerank_scores': [1.0]}

        # Build transition matrix in CSR format for efficiency
        rows, cols, data = [], [], []
        out_deg = np.array([len(neigh) for neigh in adj], dtype=np.int32)
        for u, neigh in enumerate(adj):
            if neigh:  # non‑damped nodes
                rows.extend([u]*len(neigh))
                cols.extend(neigh)
                data.extend([1.0/len(neigh)]*len(neigh))
        M = np.zeros((n, n), dtype=np.float64)
        M[rows, cols] = data

        # Handle dangling nodes (out_deg == 0)
        dangling = (out_deg == 0).astype(float)
        # Personalization vector (uniform)
        e = np.full(n, 1.0 / n)

        # Power iteration
        r = np.full(n, 1.0 / n)
        for _ in range(self.max_iter):
            r_next = self.alpha * (M @ r + dangling @ r / n) + (1 - self.alpha) * e
            if np.linalg.norm(r_next - r, 1) < self.tol:
                r = r_next
                break
            r = r_next

        return {'pagerank_scores': r.tolist()}