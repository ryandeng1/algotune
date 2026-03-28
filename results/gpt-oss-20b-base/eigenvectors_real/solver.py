import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> tuple[list[float], list[list[float]]]:
        # Compute eigenvalues and eigenvectors of a real symmetric matrix.
        eigvals, eigvecs = np.linalg.eigh(problem)
        # Reverse to descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        # Convert to Python lists
        return (eigvals.tolist(), [col.tolist() for col in eigvecs.T])