import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem["matrix"]
        k = problem["n_components"]
        n, m = A.shape
        
        Omega = np.random.randn(m, k)
        Y = A.T @ Omega
        Q, R = np.linalg.qr(Y, mode='reduced')
        U_svd, s, V_svd = np.linalg.svd(R, full_matrices=False)
        U = Q @ U_svd
        V = V_svd.T
        
        return {"U": U, "S": s, "V": V}
