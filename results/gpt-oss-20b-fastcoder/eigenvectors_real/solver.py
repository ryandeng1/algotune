import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        eigvals, eigvecs = np.linalg.eigh(problem)
        eigvals_desc = eigvals[::-1].tolist()
        eigvecs_desc = eigvecs[:, ::-1].T.tolist()
        return eigvals_desc, eigvecs_desc