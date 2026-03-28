import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        # Directly compute the LU factorization: A = P @ L @ U
        P, L, U = lu(A)
        # Convert the result to nested Python lists for the expected output format
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}