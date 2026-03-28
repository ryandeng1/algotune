import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem["matrix"]
        n_components = problem["n_components"]
        matrix_type = problem["matrix_type"]

        # decide number of power iterations
        n_iter = 10 if matrix_type == "ill_conditioned" else 5

        # if number of components is larger than or equal to the rank, use full SVD
        m, n = A.shape
        min_dim = min(m, n)
        if n_components >= min_dim:
            U, s, Vt = svd(A, full_matrices=False)
            U = U[:, :n_components]
            s = s[:n_components]
            Vt = Vt[:n_components, :]
        else:
            # use sparse SVD with power iterations
            # svds does not expose n_iter, so we implement a simple power iteration wrapper
            def power_iteration(A, k, n_iter, rng):
                # initial random matrix
                X = rng.standard_normal((A.shape[1], k))
                for _ in range(n_iter):
                    X = A @ (A.T @ X)
                    X, _ = np.linalg.qr(X)
                return X

            rng = np.random.default_rng(42)
            # orthonormal bases for power iterations
            X = power_iteration(A, n_components, n_iter, rng)
            B = A @ X
            U_hat, s, Vt = svds(A, k=n_components, return_singular_vectors=True)
            # svds returns in arbitrary order, sort
            order = np.argsort(s)[::-1]
            s = s[order]
            U_hat = U_hat[:, order]
            Vt = Vt[order, :]

            U = U_hat
        return {"U": U, "S": s, "V": Vt.T}