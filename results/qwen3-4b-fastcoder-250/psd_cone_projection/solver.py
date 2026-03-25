import numpy as np
from scipy import linalg as la

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        A = np.array(problem["A"])
        eigvals, eigvecs = la.eigh(A)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return {"X": X}
