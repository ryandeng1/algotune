from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem["matrix"]
        n_components = problem["n_components"]
        # Use the deterministic SVD provided by NumPy for speed.
        U, s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        U = U[:, :n_components]
        s = s[:n_components]
        Vt = Vt[:n_components, :]
        return {"U": U, "S": s, "V": Vt.T}