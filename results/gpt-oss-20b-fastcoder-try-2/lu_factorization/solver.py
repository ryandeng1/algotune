import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        # Compute LU factorization P L U of the input matrix A
        A = problem["matrix"]
        P, L, U = lu(A)

        # Convert the resulting matrices to Python lists for the required output format
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}