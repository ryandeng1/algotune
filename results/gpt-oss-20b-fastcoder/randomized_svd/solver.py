import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem['matrix']
        n_components = problem['n_components']

        # Compute reduced SVD: U[:, :k], s[:k], Vt[:k, :]
        # Use np.linalg.svd with full_matrices=False for speed
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # Truncate to the requested number of components if needed
        if U.shape[1] > n_components:
            U = U[:, :n_components]
            s = s[:n_components]
            Vt = Vt[:n_components, :]

        return {
            'U': U,
            'S': s,
            'V': Vt.T
        }