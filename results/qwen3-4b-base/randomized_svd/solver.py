import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem["matrix"]
        n_components = problem["n_components"]
        n = problem["n"]
        m = problem["m"]
        matrix_type = problem.get("matrix_type", "general")
        n_iter = 10 if matrix_type == "ill_conditioned" else 5
        
        Omega = np.random.randn(m, n_components)
        B = np.dot(A, Omega)
        U_B, S_B, V_B = np.linalg.svd(B, full_matrices=False)
        
        U = np.dot(A, V_B) / S_B
        S = S_B
        V = V_B
        
        return {"U": U, "S": S, "V": V}
