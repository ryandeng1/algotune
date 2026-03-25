import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> np.ndarray:
        A = problem["matrix"]
        A_dense = A.toarray()
        expA_dense = expm(A_dense)
        return csc_matrix(expA_dense)
