from typing import Any
from scipy import sparse
from scipy.sparse.linalg import expm


class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        """
        Computes the matrix exponential of the sparse matrix `problem["matrix"]`.
        The function coerces the input to CSR format if necessary (the most efficient
        format for the underlying implementation) and then calls
        `scipy.sparse.linalg.expm` to perform the computation.

        Args:
            problem: A dictionary with a key `"matrix"` whose value is a
                     `scipy.sparse.spmatrix` representing the matrix to exponentiate.

        Returns:
            The matrix exponential as a sparse matrix.
        """
        A = problem["matrix"]
        # Ensure the matrix is in CSR format for optimal performance.
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()
        return expm(A)