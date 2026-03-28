import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        # Extract the matrix
        A = problem["matrix"]
        # Compute the LU decomposition with partial pivoting
        P, L, U = lu(A)
        # Convert the resulting matrices to plain Python lists
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}