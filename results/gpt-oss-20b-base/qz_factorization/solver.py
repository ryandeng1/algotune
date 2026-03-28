from typing import Any, Dict, List
import numpy as np
from scipy.linalg import qz


class Solver:
    def solve(
        self,
        problem: Dict[str, List[List[float]]]
    ) -> Dict[str, Dict[str, List[List[float | complex]]]]:
        """
        Compute the QZ decomposition of the matrix pair (A, B) using the fast
        SciPy implementation.  The input matrices are converted to NumPy arrays
        with the minimal cast needed, and the result is returned as a plain
        nested‑list structure suitable for JSON serialisation.

        Parameters
        ----------
        problem
            A dictionary containing the input matrices under the keys ``'A'`` and
            ``'B'``.  Each entry is a list of lists of floats.

        Returns
        -------
        dict
            A dictionary ``{'QZ': {'AA': ..., 'BB': ..., 'Q': ..., 'Z': ...}}``.
        """
        # Convert input to contiguous float64 arrays once – this is the most
        # expensive step, but it is unavoidable.  Using ``dtype`` forces a
        # standard layout that SciPy’s implementation can use directly.
        A = np.asarray(problem["A"], dtype=np.float64, order="C")
        B = np.asarray(problem["B"], dtype=np.float64, order="C")

        # SciPy’s ``qz`` operates on float64 matrices and returns float64
        # matrices for ``output='real'``.  The computation is done in place
        # on the provided arrays, so we avoid any unnecessary copies.
        AA, BB, Q, Z = qz(A, B, output="real")

        # ``tolist`` is the fastest way to produce a JSON‑friendly representation.
        # The lists are created lazily and are small compared to the matrices
        # themselves, so the overhead is negligible.
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }