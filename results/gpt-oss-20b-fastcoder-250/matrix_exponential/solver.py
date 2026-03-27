from typing import Dict, List
import numpy as np
from scipy.linalg import expm


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Compute the matrix exponential of the input matrix ``A``.

        Parameters
        ----------
        problem
            A mapping that contains the key ``"matrix"`` pointing to a NumPy array.

        Returns
        -------
        dict
            A mapping with the key ``"exponential"`` whose value is a list-of-lists representation
            of e^A.  Converting to a plain list keeps the output compatible with the API
            expectations while avoiding unnecessary copies during the calculation.
        """
        A = problem["matrix"]
        # Direct call to the highly‑optimized SciPy implementation
        expA = expm(A)
        # Convert the resulting NumPy array to a plain list of lists
        return {"exponential": expA.tolist()}