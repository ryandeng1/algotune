import numpy as np

class Solver:
    def solve(self, problem):
        """
        Compute the singular value decomposition of a matrix A using NumPy.
        The function returns the matrices as Python lists to match the expected
        output format.
        """
        # Extract matrix as a NumPy array (copy avoided if possible)
        A = np.asarray(problem["matrix"], dtype=float)

        # Perform compact SVD
        # Use the 'auto' Jacobi algorithm which is fast for dense matrices
        U, s, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)

        # Transpose Vh to obtain V
        V = Vh.T

        # Convert NumPy arrays to plain Python lists
        return {
            "U": U.tolist(),
            "S": s.tolist(),
            "V": V.tolist()
        }