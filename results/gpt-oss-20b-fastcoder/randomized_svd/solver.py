import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem['matrix']
        n_components = problem['n_components']
        # Perform a full SVD (efficient when matrix size is moderate)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Return only the first n_components components
        return {
            'U': U[:, :n_components],
            'S': s[:n_components],
            'V': Vt[:n_components, :].T
        }