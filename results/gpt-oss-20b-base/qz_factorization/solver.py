import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Compute the real QZ factorization of a matrix pair (A, B).

        Parameters
        ----------
        problem : dict
            Must contain keys 'A' and 'B', each mapped to a list of lists of floats
            representing the matrices.

        Returns
        -------
        dict
            A dictionary with a single key 'QZ'.  The value is another dictionary with
            the keys 'AA', 'BB', 'Q', and 'Z', each containing the corresponding
            matrix converted to a list-of-lists representation.

        Notes
        -----
        - The function relies on :func:`scipy.linalg.qz`, which is highly
          optimized and written in C.  No additional Python loops or
          expensive operations are performed.
        - All intermediate arrays remain NumPy ``ndarray`` objects until the very
          last moment, where ``tolist`` is called once per output matrix.
        """
        A = np.asarray(problem['A'], dtype=np.float64, order='C')
        B = np.asarray(problem['B'], dtype=np.float64, order='C')

        # Perform the real QZ factorization
        AA, BB, Q, Z = qz(A, B, output='real')

        # Packaging the result
        return {
            'QZ': {
                'AA': AA.tolist(),
                'BB': BB.tolist(),
                'Q': Q.tolist(),
                'Z': Z.tolist()
            }
        }