from typing import Any, Tuple, List
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the eigenvalue problem for a real symmetric matrix.

        Returns a tuple (eigenvalues, eigenvectors) where:
        - eigenvalues is a list of floats sorted in descending order.
        - eigenvectors is a list of lists, with each inner list an eigenvector
          normalized to unit length, sorted corresponding to the eigenvalues.
        """
        # Compute eigenvalues and eigenvectors (ascending order) in one call,
        # then reverse both for descending order.
        vals, vecs = np.linalg.eigh(problem)
        vals = np.flip(vals)
        vecs = np.flip(vecs, axis=1)

        # Convert to plain Python lists efficiently.
        return vals.tolist(), vecs.T.tolist()