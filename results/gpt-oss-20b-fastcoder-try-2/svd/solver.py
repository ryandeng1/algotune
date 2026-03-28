import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Fast SVD using NumPy's optimized routine.
        """
        A = problem['matrix']
        # NumPy's linalg.svd is highly tuned for the underlying BLAS.
        U, s, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        # No transpose needed when returning raw NumPy arrays; callers can handle shape.
        return {'U': U, 'S': s, 'V': Vh}