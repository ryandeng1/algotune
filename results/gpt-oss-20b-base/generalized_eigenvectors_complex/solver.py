import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        A, B = problem
        # scale for numerical stability
        scl = np.sqrt(np.linalg.norm(B, ord='fro'))
        A_s = A / scl
        B_s = B / scl
        # solve without validity checks to save time
        w, v = la.eig(A_s, B_s, check_finite=False)
        # normalize eigenvectors in place
        norms = np.linalg.norm(v, axis=0)
        mask = norms > 1e-15
        v[:, mask] /= norms[mask]
        # sort eigenpairs
        idx = np.lexsort((-w.real, -w.imag))
        w = w[idx]
        v = v[:, idx]
        # convert to Python lists
        return (w.tolist(), [vec.tolist() for vec in v.T])