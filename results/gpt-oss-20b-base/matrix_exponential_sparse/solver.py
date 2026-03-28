from scipy.sparse import isspmatrix_csr
from scipy.sparse.linalg import expm

class Solver:
    def solve(self, problem: dict[str, "scipy.sparse.spmatrix"]) -> "scipy.sparse.spmatrix":
        A = problem["matrix"]
        # Ensure CSR format for fast exponentiation
        if not isspmatrix_csr(A):
            A = A.tocsr()
        return expm(A)