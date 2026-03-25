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
        
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvalues = np.real(eigenvalues)
        exp_eigenvalues = np.exp(eigenvalues)
        expA = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
        
        result_comm = {}
        for i in range(n):
            result_comm[i] = {}
            for j in range(n):
                result_comm[i][j] = float(expA[i, j])
        
        return {"communicability": result_comm}
