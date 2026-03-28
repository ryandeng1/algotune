import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        A = problem["matrix"]
        n_components = problem["n_components"]
        matrix_type = problem.get("matrix_type", None)

        # Use different number of power iterations for ill‑conditioned matrices
        n_iter = 10 if matrix_type == "ill_conditioned" else 5

        # Set deterministic random seed
        rng = np.random.default_rng(42)

        # Step 1: Random projection
        n_features = A.shape[1]
        random_matrix = rng.standard_normal((n_features, n_components))
        Y = A @ random_matrix

        # Step 2: Repeated QR to form an orthonormal basis for the Krylov subspace
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(Y, mode="reduced")
            Y = A @ (Q @ random_matrix.T)

        Q, _ = np.linalg.qr(Y, mode="reduced")

        # Step 3: Project A onto the subspace
        B = Q.T @ A

        # Step 4: Thin SVD of the small matrix B
        U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)

        # Step 5: Lift U_hat back to the original space
        U = Q @ U_hat

        return {"U": U, "S": s, "V": Vt.T}