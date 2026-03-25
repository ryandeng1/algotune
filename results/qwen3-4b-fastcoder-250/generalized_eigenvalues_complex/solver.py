import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> list[complex]:
        A, B = problem
        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B
        eigenvalues, _ = la.eig(A_scaled, B_scaled)
        return sorted(eigenvalues, key=lambda x: (-x.real, -x.imag))
