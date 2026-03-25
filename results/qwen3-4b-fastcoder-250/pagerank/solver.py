import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
        
        alpha = 0.85
        
        deg = [len(adj) for adj in adj_list]
        
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            d_i = deg[i]
            if d_i > 0:
                for j in adj_list[i]:
                    rows.append(j)
                    cols.append(i)
                    data.append(1.0 / d_i)
            else:
                for j in range(n):
                    rows.append(j)
                    cols.append(i)
                    data.append(1.0 / n)
        
        P = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        I = csr_matrix(np.eye(n))
        A = I - alpha * P
        
        b = (1.0 - alpha) / n * np.ones(n)
        
        r = spsolve(A, b)
        
        return {"pagerank_scores": r.tolist()}
