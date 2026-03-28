import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        """
        Compute the product matrix C = A · B using NumPy for optimal speed.
        """
        # Directly convert input lists to NumPy arrays with float64 precision
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        # Perform matrix multiplication
        C = A @ B  # equivalent to np.dot(A, B) but slightly clearer

        # Return a plain Python list of lists
        return C.tolist()