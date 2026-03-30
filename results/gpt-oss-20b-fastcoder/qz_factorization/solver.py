# solver.py

from typing import Any, Dict, List
import numpy as np
from scipy.linalg import qz


class Solver:
    """
    Solver for the QZ factorization problem.

    The implementation is deliberately minimal and relies on SciPy's
    highly optimized routine.  Mostly the focus is on avoiding any
    unnecessary data copies and on keeping the code simple and
    maintainable.
    """

    def solve(
        self,
        problem: Dict[str, List[List[float]]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Compute the real generalized Schur decomposition (QZ),
        i.e. find matrices Q, Z, AA, BB such that

            A = Q @ AA @ Z.T
            B = Q @ BB @ Z.T

        Parameters
        ----------
        problem : dict
            Must contain two keys:
                'A' : [[float]] – first matrix
                'B' : [[float]] – second matrix

        Returns
        -------
        dict
            Mapping {'QZ': {'AA': ..., 'BB': ..., 'Q': ..., 'Z': ...}}
            where every entry is converted to the native Python list
            representation expected by the evaluation harness.
        """
        # 1. Create NumPy arrays without copying data if possible
        A = np.asarray(problem["A"], dtype=np.float64, order="C")
        B = np.asarray(problem["B"], dtype=np.float64, order="C")

        # 2. Perform the QZ factorization in real mode
        AA, BB, Q, Z = qz(A, B, output="real")

        # 3. Convert the result to plain Python lists for serialisation
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist(),
            }
        }