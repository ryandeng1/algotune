from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        A = problem["A"]
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return {"X": X}