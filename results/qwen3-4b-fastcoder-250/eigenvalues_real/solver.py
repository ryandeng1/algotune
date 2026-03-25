import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> list[float]:
        eigenvalues = np.linalg.eigh(problem)[0]
        return eigenvalues[::-1].tolist()
