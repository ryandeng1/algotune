import numpy as np
from scipy.sparse import csr_matrix

class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-5

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        d = np.array([len(adj_list[i]) for i in range(n)])
        
        rows = []
        cols = []
        data = []
        for i in range(n):
            if d[i] > 0:
                for j in adj_list[i]:
                    rows.append(j)
                    cols.append(i)
                    data.append(1.0 / d[i])
            else:
                for j in range(n):
                    rows.append(j)
                    cols.append(i)
                    data.append(1.0 / n)
        
        P = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        r = np.ones(n) / n
        r_prev = r.copy()
        
        for _ in range(self.max_iter):
            r = self.alpha * P.dot(r) + (1 - self.alpha) * (np.ones(n) / n)
            if np.linalg.norm(r - r_prev, ord=1) < self.tol:
                break
            r_prev = r.copy()
        
        return {"pagerank_scores": r.tolist()}
