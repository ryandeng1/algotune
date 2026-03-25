import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> tuple[list[float], list[list[float]]]:
        A, B = problem
        n = A.shape[0]
        
        L = np.linalg.cholesky(B)
        L_inv = np.linalg.inv(L)
        Atilde = L_inv @ A @ L_inv.T
        
        eigenvalues, eigenvectors = np.linalg.eigh(Atilde)
        
        eigenvectors = L_inv.T @ eigenvectors
        
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = [eigenvectors[:, i].tolist() for i in range(eigenvectors.shape[1])]
        
        return (eigenvalues_list, eigenvectors_list)
