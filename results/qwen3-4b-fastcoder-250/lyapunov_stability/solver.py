import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"])
        n = A.shape[0]
        
        eigenvalues = np.linalg.eigvals(A)
        if np.any(np.abs(eigenvalues) >= 1 - 1e-10):
            return {"is_stable": False, "P": None}
        
        kron_product = np.kron(A.T, A)
        K = kron_product - np.eye(n * n)
        b = -np.eye(n).reshape(-1)
        
        x = np.linalg.solve(K, b)
        P = x.reshape((n, n))
        
        return {"is_stable": True, "P": P.tolist()}
