import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in adj_list[i]:
                A[i, j] = 1.0
        
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        exp_eigenvalues = np.exp(eigenvalues)
        eA = np.dot(eigenvectors, np.dot(np.diag(exp_eigenvalues), eigenvectors.T))
        
        result = {}
        for i in range(n):
            row = {}
            for j in range(n):
                row[j] = float(eA[i, j])
            result[i] = row
        
        return {"communicability": result}
