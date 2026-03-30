# solver.py
from typing import Dict
from scipy import sparse
from scipy.sparse.linalg import expm

class Solver:
    """
    Computes the matrix exponential of a sparse matrix efficiently.
    Utilises SciPy's specialised `expm` routine which internally
    selects the most appropriate method (Pade + scaling or Taylor)
    for the given matrix structure.
    """

    def solve(self, problem: Dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        """
        Parameters
        ----------
        problem: Dict[str, sparse.spmatrix]
            Dictionary with key 'matrix' containing a sparse matrix A.

        Returns
        -------
        sparse.spmatrix
            The exponential exp(A) as a sparse matrix.
        """
        # Expect the input to be a sparse matrix; no additional conversion required.
        A = problem["matrix"]
        return expm(A)