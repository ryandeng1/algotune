import numpy as np

class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        # Compute the polynomial roots
        roots = np.roots(problem)

        # Keep only real parts when the imaginary part is negligible
        roots = np.real_if_close(roots, tol=0.001)

        # Extract real numbers and sort them in descending order
        real_roots = np.sort(roots.astype(np.float64))[::-1]

        return real_roots.tolist()