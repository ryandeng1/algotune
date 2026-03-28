import numpy as np
from numpy.typing import NDArray

def solve(problem: NDArray) -> list[complex]:
    """
    Solve the eigenvalue problem for the given square matrix.
    The solution returned is a list of eigenvalues sorted in descending order
    by real part, then by imaginary part.
    """
    vals = np.linalg.eigvals(problem)          # efficient eigenvalue computation
    # Use numpy.lexsort on negative real and imaginary parts for descending order
    ix = np.lexsort((-vals.imag, -vals.real))
    return vals[ix].tolist()