import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        y0 = np.array(problem["y0"])
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        N = params["num_points"]
        
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = -2.0 / (dx ** 2)
            if i > 0:
                A[i, i-1] = 1.0 / (dx ** 2)
            if i < N-1:
                A[i, i+1] = 1.0 / (dx ** 2)
        
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        c = np.dot(eigenvectors.T, y0)
        
        exp_term = np.exp(alpha * (t1 - t0) * eigenvalues)
        
        u = np.dot(eigenvectors, c * exp_term)
        
        return u.tolist()
