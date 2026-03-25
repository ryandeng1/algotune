import numpy as np
from numpy.typing import NDArray
from typing import List, Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[complex]:
        """
        Compute the eigenvalues of the given real square matrix and return them
        sorted in descending order by real part and then by imaginary part.

        Parameters
        ----------
        problem : NDArray
            Real-valued square matrix.

        Returns
        -------
        List[complex]
            List of eigenvalues sorted as specified.
        """
        # Compute eigenvalues using NumPy's efficient LAPACK routine
        eigs = np.linalg.eigvals(problem).astype(complex, copy=False)

        # Sort by (-real, -imag) to achieve descending order
        sorted_eigs = sorted(eigs, key=lambda z: (-z.real, -z.imag))

        return sorted_eigs
