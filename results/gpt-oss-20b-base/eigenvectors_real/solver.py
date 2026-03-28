import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        eigvals, eigvecs = np.linalg.eigh(problem)
        # Reverse to descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        return (eigvals.tolist(), eigvecs.tolist())