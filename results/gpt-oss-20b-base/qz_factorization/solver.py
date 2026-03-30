# solver.py

from typing import Dict, List, Union
import numpy as np
from scipy.linalg import qz


class Solver:
    """
    Solver for the QZ factorization problem.

    The implementation uses `scipy.linalg.qz` with ``output='real'`` for
    optimal performance, avoiding unnecessary Python list conversions until
    the very last step.  The input matrices are converted to NumPy arrays
    with the default float type – the fastest for SciPy's native code.
    """

    def solve(
        self,
        problem: Dict[str, List[List[float]]]
    ) -> Dict[str, Dict[str, List[List[Union[float, complex]]]]]:
        """
        Compute the generalized Schur (QZ) factorization of a matrix pair (A, B).

        Parameters
        ----------
        problem : dict
            Dictionary with keys ``'A'`` and ``'B'`` mapping to 2‑D lists
            of floats that will be converted to NumPy arrays.

        Returns
        -------
        dict
            Dictionary with a single key ``'QZ'`` containing the matrices
            ``'AA'`` (generalized Schur form of A), ``'BB'`` (upper triangular
            form of B), ``'Q'`` and ``'Z'`` (unitary matrices).  Values are
            returned as plain Python lists for interoperability.
        """
        # Convert input to NumPy arrays once; SciPy works directly on these.
        A = np.array(problem["A"], dtype=np.float64, copy=False)
        B = np.array(problem["B"], dtype=np.float64, copy=False)

        # Perform the QZ factorization.
        AA, BB, Q, Z = qz(A, B, output="real")

        # Only convert to Python lists when building the result.
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }