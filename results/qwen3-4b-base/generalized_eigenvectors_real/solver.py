import numpy as np
from numba import njit

@njit
def normalize_eigenvectors(eigenvectors, B):
    n, m = eigenvectors.shape
    result = np.zeros((n, m), dtype=np.float64)
    for i in range(m):
        v = eigenvectors[:, i]
        Bv = B @ v
        norm = np.sqrt(np.dot(v, Bv))
        if norm > 0:
            result[:, i] = v / norm
    return result

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> tuple[list[float], list[list[float]]]:
        A, B = problem
        n = A.shape[0]
        
        L = np.linalg.cholesky(B)
        Linv = np.linalg.inv(L)
        
        Atilde = Linv @ A @ Linv.T
        
        eigenvalues, eigenvectors = np.linalg.eigh(Atilde)
        
        eigenvectors = Linv.T @ eigenvectors
        
        normalized_eigenvectors = normalize_eigenvectors(eigenvectors, B)
        
        eigenvalues = eigenvalues[::-1]
        normalized_eigenvectors = normalized_eigenvectors[:, ::-1]
        
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = [normalized_eigenvectors[:, i].tolist() for i in range(normalized_eigenvectors.shape[1])]
        
        return (eigenvalues_list, eigenvectors_list)
