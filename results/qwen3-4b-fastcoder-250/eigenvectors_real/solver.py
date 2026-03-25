import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> tuple[list[float], list[list[float]]]:
        eigenvalues, eigenvectors = np.linalg.eigh(problem)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        return (eigenvalues.tolist(), [eigenvectors[:, i].tolist() for i in range(eigenvectors.shape[1])])
